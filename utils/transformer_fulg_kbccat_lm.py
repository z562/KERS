from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_utils import SequenceSummary, BeamHypotheses
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import logging
from torch.nn import functional as F
from transformers import BertModel
from .encoder_pln import EncoderModel
from .decoder_fulg_kbccat import DecoderModel
import copy

logger = logging.getLogger(__name__)


class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0, reduce=None):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduce=reduce)
        else:
            self.criterion = nn.NLLLoss(reduce=reduce)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


class MultiTargetModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config_e = config
        config_d = copy.copy(config)
        config_e.n_layer = 6
        config_d.n_layer = 6

        self.dev = torch.device("cuda")
        self.encoder = EncoderModel(config_e)
        self.decoder = DecoderModel(config_d)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.gen = nn.Sequential(nn.Dropout(0.1),
                                     nn.Linear(config.n_embd, 1, bias=False),
                                     nn.Sigmoid())

        self.init_weights()

        #self.input_pre_embs()

    def input_pre_embs(self):
        vocab = []
        with open("embed.txt", encoding='utf-8') as f:
            for _, line in enumerate(f.readlines()):
                if len(line.split('\t')) < 2:
                    vocab.append(self.encoder.wte.weight[_].tolist())
                else:
                    emb = line.split('\t')[-1].split(' ')
                    emb = list(map(float, emb))
                    vocab.append(emb)
        self.encoder.wte = nn.Embedding.from_pretrained(torch.FloatTensor(vocab))
        self.lm_head.weight = self.encoder.wte.weight

    def set_input_embeddings(self, new_embeddings):
        self.encoder.wte = new_embeddings

    def get_input_embeddings(self):
        return self.encoder.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        bts = input_ids[0].shape[0]

        his_ids = input_ids[0]
        res_ids = input_ids[1]
        kn_ids = input_ids[2]
        voc_ids = input_ids[3]
        goal_ids = input_ids[4]
        cls = 7 * torch.ones([bts, 1]).long().to(self.dev)

        his_mask = his_ids != 0
        his_mask = his_mask.long()
        kn_mask = kn_ids != 0
        kn_mask = kn_mask.long()
        goal_mask = goal_ids != 0
        goal_mask = goal_mask.long()
        cls_mask = cls != 0
        cls_mask = cls_mask.long()

        his_ids = torch.cat([cls, his_ids], -1)
        token_type_ids = torch.cat([cls, token_type_ids], -1)
        his_mask = torch.cat([cls_mask, his_mask], -1)

        his_outputs = self.encoder(
            input_ids=his_ids,
            past=past,
            attention_mask=his_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        goal_outputs = self.encoder(
            input_ids=goal_ids,
            past=past,
            attention_mask=goal_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        kn_outputs = self.encoder(
            input_ids=kn_ids,
            past=past,
            attention_mask=kn_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        res_embs = self.encoder.wte(res_ids)
        if position_ids is None:
            device = res_embs.device
            position_ids = torch.arange(0, res_embs.shape[-2] + 0, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, res_embs.shape[-2])
        position_embeds = self.encoder.wpe(position_ids)
        res_embs = res_embs + position_embeds
        decoder_outputs = self.decoder(
            inputs_embeds=res_embs,
            context=his_outputs[0],
            knowledge=kn_outputs[0],
            comment=his_outputs[0],
            goal=goal_outputs[0],
            past=past,
            his_attention_mask=his_mask,
            kn_attention_mask=kn_mask,
            com_attention_mask=None,
            goal_attention_mask=goal_mask,
            head_mask=head_mask,
        )

        hidden_states = decoder_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        gen_p = self.gen(hidden_states).view(bts * hidden_states.shape[1], -1)
        tmp_gen_p = gen_p.repeat(1, voc_ids.shape[-1])
        tmp_lm = lm_logits.view(bts * lm_logits.shape[1], -1)
        tmp_voc = voc_ids.view(bts, 1, voc_ids.shape[-1]).repeat(1, lm_logits.shape[1], 1).view(bts * lm_logits.shape[1], -1)
        kn_p = torch.zeros(tmp_lm.shape).to(self.dev).scatter_(1, tmp_voc, 1 - tmp_gen_p)
        kn_p[:, 0] = 0.0
        lm_logits = gen_p * tmp_lm + kn_p * tmp_lm
        lm_logits = lm_logits.view(bts, hidden_states.shape[1], -1)

        mc_logits = self.multiple_choice_head(his_outputs[0], mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + decoder_outputs[1:]
        # if use_bow:
        #     bow_logits = self.bow_fc(knowledge)
        #     bow_logits = self.bow_lm(bow_logits).repeat(1, hidden_states.shape[1], 1)
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(bow_logits.view(-1, bow_logits.size(-1)), lm_labels.view(-1))
        #     outputs = (loss,) + outputs

        # if use_posterior:
        #     prior_attn = torch.log(x_kn_att + 1e-10).view(bts, -1)
        #     posterior_att = torch.FloatTensor(xy_kn_att.tolist()).to(self.device).view(bts, -1)
        #     kl_loss = posterior_att * (torch.log(posterior_att + 1e-10) - prior_attn)
        #     kl_loss = torch.mean(kl_loss)
        #     outputs = (kl_loss,) + outputs

        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            shift_labels[shift_labels == -100] = 0
            # loss_fct = NMTCritierion(reduce=False)
            # loss_fct_3 = CrossEntropyLoss(reduce=False)
            loss_fct = NMTCritierion(0.2, reduce=False)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = torch.sum(loss, -1)
            loss = torch.sum(loss * (shift_labels != 0).view(-1).float()) / torch.sum((shift_labels != 0).view(-1).float())
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss = torch.sum(loss * (shift_labels != 0).view(-1)) / torch.sum((shift_labels != 0).view(-1))
            # loss_3 = loss_fct_3(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss_3 = torch.sum(loss_3 * (shift_labels != 0).view(-1)) / torch.sum((shift_labels != 0).view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False

    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            max_length=None,
            do_sample=True,
            num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_ids=None,
            length_penalty=None,
            num_return_sequences=None,
            knowledges=None,
            kn_vocs=None,
            segments=None,
            goals=None,
    ):
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_ids is None) or (
                isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        if pad_token_id is None and eos_token_ids is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_ids[0])
            )
            pad_token_id = eos_token_ids[0]

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
                knowledges=knowledges,
                kn_vocs=kn_vocs,
                segments=segments,
                goals=goals
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                knowledges=knowledges,
                kn_vocs=kn_vocs,
                segments=segments,
                goals=goals
            )

        return output

    def _generate_no_beam_search(
            self,
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            batch_size,
            knowledges=None,
            kn_vocs=None,
            segments=None,
            goals=None,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length - input_ids.shape[1])

        past = None
        device = torch.device("cuda")
        responses = 3 * torch.ones([input_ids.shape[0], 1]).long().to(device)

        while cur_len < max_length:
            model_inputs = {"input_ids": (input_ids, responses, knowledges, kn_vocs, goals), "token_type_ids": segments}
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            responses = torch.cat([responses, tokens_to_add.unsqueeze(-1)], dim=-1)
            # print(pro)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()) != 0
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1 - input_ids.shape[-1])
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = responses.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = responses

        for hypo_idx, hypo in enumerate(responses):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
        knowledges=None,
        kn_vocs=None,
        segments=None,
        goals=None,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            scores = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
                next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item()
                )
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_ids is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # add to generated hypotheses if end of sentence or last iteration
                    if eos_token_ids is not None and word_id.item() in eos_token_ids:
                        generated_hyps[batch_idx].add(
                            input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        # add next predicted word if it is not eos_token
                        next_sent_beam.append((score, word_id, batch_idx * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 2nd position
                    reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        for batch_idx in range(batch_size):
            # Add all open beam hypothesis to generated_hyps
            if not done[batch_idx]:
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size
                    generated_hyps[batch_idx].add(
                        input_ids[batch_idx * num_beams + beam_id, :cur_len].clone(), score.item()
                    )

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.beams, key=lambda x: x[0])[1]
            sent_lengths[i] = len(best_hyp)
            best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_ids[0]
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



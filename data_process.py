import numpy as np
import os
import json
import re
from pprint import pprint
import pickle
import math
from analyze import *

train_path = 'data/train'


def rebuildvoc():
    with open("vocab.txt", encoding="utf-8") as f:
        voc = [word.strip().split('\t')[0] for word in f.readlines()]

    sp_token = 0
    for line in voc:
        if line[0] == '<':
            sp_token += 1
        else:
            break

    mask_list = {}
    names = getusername()
    highes = getmaskword('身高')
    weights = getmaskword('体重')
    constells = getmaskword('星座')
    bloods = getmaskword('血型') - set('O')

    for name in names:
        mask_list.update({name: '<name>'})
    for high in highes:
        mask_list.update({high: '<high>'})
    for weight in weights:
        mask_list.update({weight: '<weight>'})
    for constell in constells:
        mask_list.update({constell: '<constell>'})
    for blood in bloods:
        mask_list.update({blood: '<blood>'})

    for name in names:
        voc.remove(name)
    voc.insert(sp_token, '<name>')
    sp_token += 1

    for high in highes:
        voc.remove(high)
    voc.insert(sp_token, '<high>')
    sp_token += 1

    for weight in weights:
        voc.remove(weight)
    voc.insert(sp_token, '<weight>')
    sp_token += 1

    for constell in constells:
        voc.remove(constell)
    voc.insert(sp_token, '<constell>')
    sp_token += 1

    for blood in bloods:
        voc.remove(blood)
    voc.insert(sp_token, '<blood>')
    sp_token += 1

    for i in range(30):
        voc.insert(sp_token, '<goal' + str(i) + '>')
        sp_token += 1

    with open("vocab_EG.txt", "w", encoding='utf-8') as f:
        f.write('\n'.join(voc))

    return mask_list


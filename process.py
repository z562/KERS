import jieba
from collections import defaultdict
import json
import numpy as np
import os


def retrieve_knowledges(context, data, star):
    key = ['日期', '主演', '类型', '人均', '地址', '导演', '演唱', '出生地',
           '口碑', '属相', '身高', '体重', '星座', '血型', '生日', '评分']
    if len(context) == 0:
        return ['姓名 ' + data['user_profile']['姓名']], key

    q = context[-1]

    answers = []
    if ('几号' in q and '生日' not in q) or ('日子' in q and '什么' in q):
        for kb in data['knowledge']:
            if '日期' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '主演' in q and '谁' in q:
        for kb in data['knowledge']:
            if '主演' in kb[1] and kb[1] not in star:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '类型' in q:
        for kb in data['knowledge']:
            if '类型' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '人均' in q or '消费' in q:
        for kb in data['knowledge']:
            if '人均' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '地址' in q:
        for kb in data['knowledge']:
            if '地址' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '评分' in q:
        for kb in data['knowledge']:
            if '评分' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '导' in q and '谁' in q:
        for kb in data['knowledge']:
            if '导演' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '演唱 ' in q or '主唱' in q:
        for kb in data['knowledge']:
            if '演唱' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '口碑 ' in q:
        for kb in data['knowledge']:
            if '口碑' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if ' 属 ' in q or '属相' in q:
        for kb in data['knowledge']:
            if '属相' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '身高' in q or '多高' in q:
        for kb in data['knowledge']:
            if '身高' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '体重' in q or '多重' in q:
        for kb in data['knowledge']:
            if '体重' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '星座' in q:
        for kb in data['knowledge']:
            if '星座' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '血型' in q:
        for kb in data['knowledge']:
            if '血型' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '生日' in q:
        for kb in data['knowledge']:
            if '生日' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '出生地' in q:
        for kb in data['knowledge']:
            if '出生地' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '国家' in q:
        for kb in data['knowledge']:
            if '国家地区' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    if '天气' in q:
        for kb in data['knowledge']:
            if '-' in kb[1]:
                kb = ' '.join(kb)
                answers.append(' '.join(kb.split()))

    return answers, key


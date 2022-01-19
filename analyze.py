import os
import json
import numpy as np
from collections import defaultdict
from pprint import pprint
from process import retrieve_knowledges


def getusername():
    dirs = ['data/paper_test', 'data/train', 'data/test_B']
    names = set()
    for dir in dirs:
        with open(os.path.join(dir, 'data.txt'), encoding='utf-8') as f:
            for line in f.readlines():
                info = json.loads(line, encoding='utf-8')
                names.add(info['user_profile']['姓名'])
    return names


def getmaskword(attr):
    dirs = ['data/paper_test', 'data/train']
    words = set()
    for dir in dirs:
        with open(os.path.join(dir, 'data.txt'), encoding='utf-8') as f:
            for line in f.readlines():
                num = 0
                info = json.loads(line, encoding='utf-8')
                for kb in info['knowledge']:
                    if attr in kb[1]:
                        num += 1
                        words.add(kb[2])
                if num > 1:
                    print("error")
    return words




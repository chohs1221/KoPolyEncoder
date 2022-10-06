import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


def parse_data(filename):
    context, candidate = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        dataset = data[i]['utterances'][-1]

        context += dataset['history'][0:len(dataset['history']):2]
        candidate += dataset['history'][1::2]
        candidate.append(dataset['candidates'][-1])

        assert len(dataset['history']) % 2 == 1
        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


def make_testset(filename):
    context, candidate = [], []
    with open(filename, 'r') as f:
        datasets = f.readlines()
        for dataset in datasets:
            context += [dataset.strip().split('\t')[0] for _ in range(20)]
            candidate += dataset.strip().split('\t')[3].split('|')

        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_train.json")
    valid_context, valid_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_valid.json")
    test_context, test_candidate = make_testset("./data/original_data/personachat/test_none_original.txt")

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(f"test: {len(test_context) // 20}")
    pickling(f'./data/pickles/data/persona_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/data/persona_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/data/persona_test_{len(test_context) // 20}.pickle', act = 'save', data = (test_context, test_candidate))
    
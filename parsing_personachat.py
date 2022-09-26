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
        candidate += dataset['candidates'][-1]

        assert len(dataset['history']) % 2 == 1
        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


def make_testset(filename):
    context, candidate = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        utterances = data[i]['utterances']

        for utterance in utterances:
                        


        context += dataset['history'][0:len(dataset['history']):2]
        candidate += dataset['history'][1::2]
        candidate += dataset['candidates'][-1]

        assert len(dataset['history']) % 2 == 1
        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate



if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_train.json")
    valid_context, valid_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_valid.json")

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    pickling(f'./data/pickles/persona_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/persona_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/persona_test_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    
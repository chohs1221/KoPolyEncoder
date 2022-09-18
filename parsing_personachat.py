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
        dataset = data[i]['utterances'][-1]['history']

        if len(dataset) > 1:
            if len(dataset) % 2 == 0:
                context += dataset[0::2]
                candidate += dataset[1::2]
            elif len(dataset) % 2 == 1:
                context += dataset[0:len(dataset)-1:2]
                candidate += dataset[1::2]

    print(f'total {len(context)} datasets found')
    
    return context, candidate

def parse_data2(filename):
    context, candidate = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        dataset = data[i]['utterances'][-1]['history']

        if len(dataset) > 1:
            if len(dataset) % 2 == 0:
                print("ERROR")
                exit()
            elif len(dataset) % 2 == 1:
                context += dataset[0:len(dataset)-1:2]
                candidate += dataset[1::2]

                context += dataset[1::2]
                candidate += dataset[2::2]

    print(f'total {len(context)} datasets found')
    
    return context, candidate


if __name__ == "__main__":
    # train_context, train_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_train.json")
    # valid_context, valid_candidate = parse_data("./data/original_data/personachat/personachat_truecased_full_valid.json")
    train_context, train_candidate = parse_data2("./data/original_data/personachat/personachat_truecased_full_train.json")
    valid_context, valid_candidate = parse_data2("./data/original_data/personachat/personachat_truecased_full_valid.json")

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    pickling(f'./data/pickles/persona_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/persona_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/persona_test_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    
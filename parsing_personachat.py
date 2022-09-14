import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


def parse_data(dir):
    filenames = os.listdir(f"{dir}/personachat")
    filenames = [f"{dir}/personachat/{f}" for f in filenames if f.endswith(".json")]

    context, candidate = [], []
    for filename in filenames:
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


if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/train")

    train_context, valid_context, test_context = train_context[:100000], train_context[100000: 110000], train_context[110000:]
    train_candidate, valid_candidate, test_candidate = train_candidate[:100000], train_candidate[100000: 110000], train_candidate[110000:]

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(f"test: {len(test_context)}")
    pickling(f'./data/pickles/persona_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/persona_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/persona_test_{len(test_context)}.pickle', act = 'save', data = (test_context, test_candidate))
    
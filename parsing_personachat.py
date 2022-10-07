import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


def parse_data_(filename):
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

def parse_data(filename):
    your_context, your_candidate = [], []
    partners_context, partners_candidate = [], []
    with open(filename, 'r') as f:
        datasets = f.readlines()
        
        for dataset in datasets:
            if dataset.strip().split()[2] == 'persona:':
                if dataset.strip().split()[0] == '1':
                    your_history = ' '.join(dataset.strip().split()[1:]) + '\n'
                    partners_history = 'your '

                    flag = False
                    if partners_context:
                        partners_context.pop(-1)

                else:
                    if dataset.strip().split()[1] == 'your':
                        your_history += ' '.join(dataset.strip().split()[1:]) + '\n'
                    elif dataset.strip().split()[1] == "partner's":
                        partners_history += ' '.join(dataset.strip().split()[2:]) + '\n'
            else:
                your_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[0] + ' '
                your_context.append(your_history)
                your_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[1] + ' '
                your_candidate.append(dataset[dataset.index(' ') + 1:].strip().split('\t')[1])

                partners_history += dataset[dataset.index(' ') + 1:].strip().split('\t')[0] + ' ' + dataset[dataset.index(' ') + 1:].strip().split('\t')[1] + ' '
                partners_context.append(partners_history)
                if flag:
                    partners_candidate.append(dataset[dataset.index(' ') + 1:].strip().split('\t')[0])
                else:
                    flag = True

        partners_context.pop(-1)
        
        context = your_context + partners_context
        candidate = your_candidate + partners_candidate

        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


def make_testset_(filename):
    context, candidate = [], []
    with open(filename, 'r') as f:
        datasets = f.readlines()
        for dataset in datasets:
            context += [dataset.strip().split('\t')[0] for _ in range(20)]
            candidate += dataset.strip().split('\t')[3].split('|')

        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


def make_testset(filename):
    context, candidate = [], []
    with open(filename, 'r') as f:
        datasets = f.readlines()
        
        for dataset in datasets:
            if dataset.strip().split()[2] == 'persona:':
                if dataset.strip().split()[0] == '1':
                    history = ' '.join(dataset.strip().split()[1:]) + '\n'
                else:
                    if dataset.strip().split()[1] == 'your':
                        history += ' '.join(dataset.strip().split()[1:]) + '\n'

            else:
                history += dataset[dataset.index(' ') + 1:].strip().split('\t')[0] + ' '
                context += [history for _ in range(20)]
                history += dataset[dataset.index(' ') + 1:].strip().split('\t')[1] + ' '
                candidate += dataset.strip().split('\t')[3].split('|')

        assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/original_data/personachat/train_both_original.txt")
    valid_context, valid_candidate = parse_data("./data/original_data/personachat/valid_both_original.txt")
    test_context, test_candidate = make_testset("./data/original_data/personachat/test_both_original.txt")

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(f"test: {len(test_context) // 20}")
    pickling(f'./data/pickles/data/persona_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/data/persona_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/data/persona_test_{len(test_context) // 20}.pickle', act = 'save', data = (test_context, test_candidate))
    
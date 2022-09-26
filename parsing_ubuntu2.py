import csv

from utils import pickling


def parse_data(filename):
    context, candidate = [], []

    with open(filename, 'r', encoding='utf-8') as f:
        dataset = csv.reader(f)

        for i, data in enumerate(dataset):
            if i == 0:
                continue
            context.append(data[0])
            candidate.append(data[1])
    
    assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


def make_testset(filename):
    context, candidate = [], []

    with open(filename, 'r', encoding='utf-8') as f:
        dataset = csv.reader(f)

        for i, data in enumerate(dataset):
            if i == 0:
                continue
            context += [data[0] for _ in range(10)]
            candidate += data[1:]

    assert len(context) == len(candidate)

    print(f'total {len(context)} datasets found')
    
    return context, candidate


if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/original_data/ubuntu2/train.csv")
    valid_context, valid_candidate = parse_data("./data/original_data/ubuntu2/valid.csv")
    test_context, test_candidate = make_testset("./data/original_data/ubuntu2/test.csv")

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(f"test: {len(test_context)}")
    pickling(f'./data/pickles/ubuntu2_train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/ubuntu2_valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/ubuntu2_test_{len(test_context)}.pickle', act = 'save', data = (test_context, test_candidate))
    
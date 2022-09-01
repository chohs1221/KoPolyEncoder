import os
import json
import pickle
from tqdm import tqdm

def get_filename(dir):
    filenames = os.listdir(dir)
    return [dir + f for f in filenames if f.endswith(".json")]

def parse_data(filenames):
    context, candidate = [], []
    for filename in tqdm(filenames):
        dataset = []
        with open(filename, 'r', encoding= 'utf-8') as f:
            try:
                data = json.load(f)
            except:
                print(filename)
                exit(0)
            lines = data["info"][0]["annotations"]["lines"]

            for line in lines:
                dataset.append(line["norm_text"])

            if dataset:            
                context += dataset[:-1]
                candidate += dataset[1:]

    return context, candidate

def pickling(file_name, act, data = None):
    if act == 'save':
        with open(file_name, 'wb') as fw:
            pickle.dump(data, fw)
    
    elif act == 'load':
        with open(file_name, 'rb') as fr:
            data = pickle.load(fr)
        return data


if __name__ == "__main__":
    train_filenames = get_filename("./data/train/")
    valid_filenames = get_filename("./data/valid/")

    train_context, train_candidate = parse_data(train_filenames)
    valid_context, valid_candidate = parse_data(valid_filenames)

    print(len(train_context), len(train_candidate))
    print(len(valid_context), len(valid_candidate))

    pickling('./data/pickle/train.pickle', act = 'save', data = (train_context, train_candidate))
    pickling('./data/pickle/valid.pickle', act = 'save', data = (valid_context, valid_candidate))

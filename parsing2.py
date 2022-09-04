import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


def get_filename(dir):
    filenames = os.listdir(dir)
    return [dir + f for f in filenames if f.endswith(".json")]

def parse_data(dir):
    filenames = os.listdir(dir)
    filenames = [dir + f for f in filenames if f.endswith(".json")]
    print(f'{len(filenames)} json files found')

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

            if len(dataset) > 1:
                if len(dataset) % 2 == 0:
                    context += dataset[0::2]
                    candidate += dataset[1::2]
                elif len(dataset) % 2 == 1:
                    context += dataset[0:len(dataset)-1:2]
                    candidate += dataset[1::2]
            
            assert len(context) == len(candidate)
    
    print(f'{len(context)} datasets found')

    return context, candidate


if __name__ == "__main__":
    train_context, train_candidate = parse_data("./data/train/")
    valid_context, valid_candidate = parse_data("./data/valid/")

    pickling('./data/pickles/train2.pickle', act = 'save', data = (train_context, train_candidate))
    pickling('./data/pickles/valid2.pickle', act = 'save', data = (valid_context, valid_candidate))

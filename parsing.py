import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


# 주제별 텍스트 일상 대화 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543
def parse_data(dir):
    filenames = os.listdir(f"{dir}/opendomain")
    filenames = [f"{dir}/opendomain/{f}" for f in filenames if f.endswith(".json")]
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
    
    print(f'current {len(context)} datasets found')

    #########################################################################################################

    # 용도별 목적대화 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=544
    folders = os.listdir(f"{dir}/closedomain")
    for folder in folders:
        filenames = os.listdir(f"{dir}/closedomain/{folder}")
        filenames = [f"{dir}/closedomain/{folder}/{f}" for f in filenames if f.endswith(".txt")]
        print(f'{len(filenames)} txt files found')

        for filename in tqdm(filenames):
            with open(filename, 'r', encoding= 'utf-8') as f:
                dataset = [line.strip() for line in f.readlines() if len(line) > 1]
            
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

    split_index = int(len(train_context)*0.8), int(len(train_context)*0.9)

    train_context, valid_context, test_context = train_context[:split_index[0]], train_context[split_index[0]: split_index[1]], train_context[split_index[1]:]
    train_candidate, valid_candidate, test_candidate = train_candidate[:split_index[0]], train_candidate[split_index[0]: split_index[1]], train_candidate[split_index[1]:]

    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(f"test: {len(test_context)}")
    pickling(f'./data/pickles/train_{len(train_context)}.pickle', act = 'save', data = (train_context, train_candidate))
    pickling(f'./data/pickles/valid_{len(valid_context)}.pickle', act = 'save', data = (valid_context, valid_candidate))
    pickling(f'./data/pickles/test_{len(test_context)}.pickle', act = 'save', data = (test_context, test_candidate))
    
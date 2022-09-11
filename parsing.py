import os
import json
import pickle
from tqdm import tqdm

from utils import pickling


def parse_data(dir):
    # 주제별 텍스트 일상 대화 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543
    filenames = os.listdir(f"{dir}/aihub_543")
    filenames = [f"{dir}/aihub_543/{f}" for f in filenames if f.endswith(".json")]
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
    folders = os.listdir(f"{dir}/aihub_544")
    for folder in folders:
        filenames = os.listdir(f"{dir}/aihub_544/{folder}")
        filenames = [f"{dir}/aihub_544/{folder}/{f}" for f in filenames if f.endswith(".txt")]
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
        
    print(f'current {len(context)} datasets found')

    #########################################################################################################

    # 민원(콜센터) 질의-응답 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=98
    filenames = os.listdir(f"{dir}/aihub_98")
    filenames = [f"{dir}/aihub_98/{f}" for f in filenames if f.endswith(".json")]
    print(f'{len(filenames)} json files found')

    for filename in tqdm(filenames):
        dataset = []
        with open(filename, 'r', encoding= 'cp949') as f:
            try:
                data = json.load(f)
            except:
                print(filename)
                continue
            
        for i in range(len(data)):
            del data[i]['도메인']
            del data[i]['카테고리']
            del data[i]['대화셋일련번호']
            del data[i]['화자']
            del data[i]['문장번호']
            del data[i]['고객의도']
            del data[i]['상담사의도']
            del data[i]['QA']
            del data[i]['개체명 ']
            del data[i]['용어사전']
            del data[i]['지식베이스']

            dataset.append(sorted(list(data[i].values())).pop())
        
        if len(dataset) > 1:
            if len(dataset) % 2 == 0:
                context += dataset[0::2]
                candidate += dataset[1::2]
            elif len(dataset) % 2 == 1:
                context += dataset[0:len(dataset)-1:2]
                candidate += dataset[1::2]
    
    print(f'current {len(context)} datasets found')

    #########################################################################################################

    # KLUE DST https://klue-benchmark.com/tasks/73/data/description
    filenames = os.listdir(f"{dir}/klue_dst")
    filenames = [f"{dir}/klue_dst/{f}" for f in filenames if f.endswith(".json")]
    print(f'{len(filenames)} json files found')

    for filename in tqdm(filenames):
        with open(filename, 'r', encoding= 'utf-8') as f:
            data = json.load(f)
            
        dataset = []
        for annotation in data:
            for dialogue in annotation['dialogue']:
                dataset.append(dialogue['text'])
                
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
    
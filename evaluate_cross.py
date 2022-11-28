import argparse

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader

from utils import pickling, load_tokenizer_model
from dataset_tokenizer import TokenizeDataset_CrossEncoder


def evaluate(model_path, args):
    test = pickling(f"./data/pickles/data/{args.testset}", "load")
    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device=args.device)
    model.eval()

    test_dataset = TokenizeDataset_CrossEncoder(test, tokenizer, max_length = args.max_length, return_tensors='pt', device=args.device)
    test_loader = DataLoader(test_dataset, batch_size = args.batch, shuffle = True, drop_last = True)

    accuracy, precision, recall, f1 = 0, 0, 0, 0
    labels, preds = [], []
    for i, batch in tqdm(enumerate(test_loader, start=1), total=len(test_loader)):
        with torch.no_grad():
            _, logits = model(input_ids = batch['input_ids'],
                                attention_mask = batch['attention_mask'],
                                token_type_ids = batch['token_type_ids']
                                )
            
            pred = torch.argmax(logits, dim=-1)

            preds += pred.tolist()
            labels += batch['labels'].tolist()
            
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cross')
    parser.add_argument('--path', type=str, default='cross221128_1843_bs32_ep1_data105625_ko_best0')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--testset', type=str, default='ko_cross_test_13204.pickle')
    parser.add_argument('--lang', type=str, default='ko')
    parser.add_argument('--best', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("SCORE!!")
    print(args)

    model_path =  './checkpoints/' + args.path

    accuracy, precision, recall, f1 = evaluate(model_path, args)

    print(f'{args.path}')
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}")
    print(f'============================================================\n')
    
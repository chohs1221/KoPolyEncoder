import argparse
from tqdm import tqdm

import torch

from predict import load_tokenizer_model
from utils import pickling
from data_loader import DataLoader


def recall_1C(model_path, c, test_dataset, args, device):
    test_context, test_candidate = pickling(f"./data/pickles/{test_dataset}", "load")
    tokenizer, model = load_tokenizer_model(model_path, args.m, device=device)
    model.eval()

    test_loader = DataLoader(test_context, test_candidate, tokenizer)

    cnt = 0
    for i in tqdm(range(0, len(test_loader)-c, c)):
        with torch.no_grad():
            _, dot_product = model(**test_loader[i:i+c], training=True)
        result = torch.argmax(dot_product, dim=-1) == torch.arange(len(dot_product)).to(device)
        cnt += result.sum().item()
    
    return cnt / i


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='2022_09_06_02_40_bs2_ep6')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='test_81068.pickle')
    args = parser.parse_args()
    print(args)

    device = 'cuda'
    model_path =  './checkpoints/' + args.path

    score = recall_1C(model_path, 100, args.dataset, args, device)
    print(score)
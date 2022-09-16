import argparse

from tqdm import tqdm

import torch

from predict_personachat import load_tokenizer_model
from utils import pickling
from data_loader import DataLoader


def evaluate(model_path, m, test_dataset, c, device):
    test_context, test_candidate = pickling(f"./data/pickles/{test_dataset}", "load")
    tokenizer, model = load_tokenizer_model(model_path, m, device=device)
    model.eval()

    test_loader = DataLoader(test_context, test_candidate, tokenizer, shuffle=True, return_tensors='pt', device=device)

    r_at_1 = 0
    mrr = 0
    for i in tqdm(range(0, len(test_loader)-c, c)):
        with torch.no_grad():
            _, dot_product = model(**test_loader[i:i+c])
        
        result = torch.argmax(dot_product, dim=-1) == torch.arange(len(dot_product)).to(device)
        r_at_1 += result.sum().item()

        _, indices = torch.sort(dot_product, dim=-1, descending = True)
        for idx, indice in enumerate(indices):
            rank = torch.nonzero(indice == idx, as_tuple = True)[0][0].item()
            mrr += (1 / (rank+1))
                
    r_at_1 = round(100 * r_at_1 / (i+c), 2)
    mrr = round(100 * mrr / (i+c), 2)
    
    return r_at_1, mrr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='bi220907_1256_bs512_ep20_best1')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='test_10361.pickle')
    parser.add_argument('--c', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("SCORE!!")
    print(args)

    model_path =  './checkpoints/' + args.path

    r_an_1, mrr = evaluate(model_path, args.m, args.dataset, args.c, args.device)
    
    print(f"R@1/{args.c}: {r_an_1}")
    print(f"MRR: {mrr}")
    print(f'============================================================\n')
import argparse

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader

from utils import pickling, load_tokenizer_model
from dataset_tokenizer import TokenizeDataset, TokenizeDataset_CrossEncoder


def evaluate_ko(model_path, args, c):
    test_context, test_candidate = pickling(f"./data/pickles/data/{args.testset}", "load")
    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device=args.device)
    model.eval()

    test_dataset = TokenizeDataset(test_context, test_candidate, tokenizer, return_tensors='pt', device=args.device)
    test_loader = DataLoader(test_dataset, batch_size = c, shuffle = True, drop_last = True)

    r_at_1 = 0
    mrr = 0
    for i, batch in tqdm(enumerate(test_loader, start=1), total=len(test_loader)):
        with torch.no_grad():
            _, dot_product = model(**batch)
        
            result = torch.argmax(dot_product, dim=-1) == torch.arange(len(dot_product)).to(args.device)
            r_at_1 += result.sum().item()

            _, indices = torch.sort(dot_product, dim=-1, descending = True)
            for idx, indice in enumerate(indices):
                rank = torch.nonzero(indice == idx, as_tuple = True)[0][0].item()
                mrr += (1 / (rank+1))
                
    r_at_1 = round(100*r_at_1 / (i*c), 2)
    mrr = round(100*mrr / (i*c), 2)
    
    return r_at_1, mrr


def evaluate_personachat(model_path, args, c):
    test_context, test_candidate = pickling(f"./data/pickles/data/{args.testset}", "load")
    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device=args.device)
    model.eval()

    test_dataset = TokenizeDataset(test_context, test_candidate, tokenizer, return_tensors='pt', device=args.device)
    test_loader = DataLoader(test_dataset, batch_size = c, shuffle = False, drop_last = True)

    r_at_1 = 0
    mrr = 0
    for i, batch in tqdm(enumerate(test_loader, start=1), total=len(test_loader)):
        with torch.no_grad():
            _, dot_product = model(**batch)

            # logits = torch.diagonal(dot_product * torch.eye(c).to(dot_product.device))
            logits = dot_product[0]
            if torch.argmax(logits, dim=-1).item() == c-1:
                r_at_1 += 1

            _, indices = torch.sort(logits, dim=-1, descending = True)
            rank = torch.nonzero(indices == c-1, as_tuple = True)[0][0].item()
            mrr += (1 / (rank+1))
                
    r_at_1 = round(100*r_at_1 / i, 2)
    mrr = round(100*mrr / i, 2)
    
    return r_at_1, mrr


def evaluate_ubuntu2(model_path, args, c):
    test_context, test_candidate = pickling(f"./data/pickles/data/{args.testset}", "load")
    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device=args.device)
    model.eval()

    test_dataset = TokenizeDataset(test_context, test_candidate, tokenizer, return_tensors='pt', device=args.device)
    test_loader = DataLoader(test_dataset, batch_size = c, shuffle = False, drop_last = True)

    r_at_1 = 0
    mrr = 0
    for i, batch in tqdm(enumerate(test_loader, start=1), total=len(test_loader)):
        with torch.no_grad():
            _, dot_product = model(**batch)

            logits = dot_product[0]
            if torch.argmax(logits, dim=-1).item() == 0:
                r_at_1 += 1

            _, indices = torch.sort(logits, dim=-1, descending = True)
            rank = torch.nonzero(indices == 0, as_tuple = True)[0][0].item()
            mrr += (1 / (rank+1))
                
    r_at_1 = round(100*r_at_1 / i, 2)
    mrr = round(100*mrr / i, 2)
    
    return r_at_1, mrr


def evaluate_cross(model_path, args):
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
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='bi221026_1139_bs128_ep5_data562920_ko_best1')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--testset', type=str, default='ko_test_70365.pickle')
    parser.add_argument('--lang', type=str, default='ko')
    parser.add_argument('--task', type=str, default='ko')
    parser.add_argument('--best', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("SCORE!!")
    print(args)

    model_path =  './checkpoints/' + args.path

    if args.model == 'bi' or args.model == 'poly':
        if args.task == 'ko':
            for c in [100, 20, 10]:
                r_at_1, mrr = evaluate_ko(model_path, args, c)

                print(f'{args.path}')
                print(f"R@1/{c}: {r_at_1}")
                print(f"MRR: {mrr}")
                print(f'============================================================\n')
        
        elif args.task == 'personachat':
            r_at_1, mrr = evaluate_personachat(model_path, args, 20)

            print(f'{args.path}')
            print(f"R@1/{20}: {r_at_1}")
            print(f"MRR: {mrr}")
            print(f'============================================================\n')
        
        elif args.task == 'ubuntu2':
            r_at_1, mrr = evaluate_ubuntu2(model_path, args, 10)

            print(f'{args.path}')
            print(f"R@1/{10}: {r_at_1}")
            print(f"MRR: {mrr}")
            print(f'============================================================\n')
    
    elif args.model == 'cross':
        accuracy, precision, recall, f1 = evaluate_cross(model_path, args)

        print(f'{args.path}')
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1 score: {f1}")
        print(f'============================================================\n')
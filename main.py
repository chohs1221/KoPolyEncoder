import os
import argparse
from datetime import datetime
import re

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from kobert_tokenizer import KoBERTTokenizer

from modeling import BiEncoder, PolyEncoder
from dataset_tokenizer import TokenizeDataset
from utils import seed_everything, empty_cuda_cache, pickling

# os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(args):
    print(f'============================================================')
    start_time = datetime.now()
    file_name = f"{args.model}{start_time.strftime('%Y%m%d_%H%M')[2:]}_bs{args.batch * args.accumulation}_ep{args.epoch}_data{re.sub(r'[^0-9]', '', args.trainset)}_{args.lang}"
    os.rename('./output.txt', f'./outputs/{file_name}.txt')
    print(f'File Name: {file_name}')
    print(f"START!! {start_time.strftime('%Y_%m_%d / %H_%M')}")
    print(f'model: {args.model}')
    print(f'path: {args.path}')
    print(f'trainset: {args.trainset}')
    print(f'validset: {args.validset}')
    print(f'm: {args.m}')
    print(f'seed: {args.seed}')
    print(f'epoch: {args.epoch}')
    print(f'learning rate: {args.lr}')
    print(f'batch size: {args.batch}')
    print(f'accumulation: {args.accumulation}')
    print(f'language: {args.lang}')
    print(f'description: {args.description}\n')


    seed_everything(args.seed)


    train_context, train_candidate = pickling(f'./data/pickles/{args.trainset}.pickle', 'load')
    valid_context, valid_candidate = pickling(f'./data/pickles/{args.validset}.pickle', 'load')
    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(train_context[0:5])
    print(train_candidate[0:5])
    print()
    
    
    if args.lang == "ko":
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    elif args.lang == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.model == 'bi':
        model = BiEncoder.from_pretrained(args.path)
    elif args.model == 'poly':
        model = PolyEncoder.from_pretrained(args.path, m=args.m)
    model.to('cuda')


    train_dataset = TokenizeDataset(train_context, train_candidate, tokenizer, return_tensors='pt', device='cuda')
    valid_dataset = TokenizeDataset(valid_context, valid_candidate, tokenizer, return_tensors='pt', device='cuda')

    train_loader = DataLoader(train_dataset, batch_size = args.batch, shuffle = True, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch, shuffle = True, drop_last = True)


    train_loss = 0
    valid_loss = 0
    pre_valid_loss = 9999.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4)
    model.train()
    empty_cuda_cache()
    for epoch in range(args.epoch):
        with tqdm(train_loader, unit="batch") as t:
            for iteration, batch in enumerate(t, start=1):
                t.set_description(f"Epoch {epoch}")

                loss, _  = model(**batch)
                                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                
                t.set_postfix(loss=loss.item())

                if iteration % int(len(train_loader) / 10) == 0:
                    model.eval()
                    with torch.no_grad():
                        for i, batch in enumerate(valid_loader, start=1):
                            loss, _  = model(**batch)

                            valid_loss += loss.item()
                        
                        train_loss /= int(len(train_loader) / 10)
                        valid_loss /= i
                    
                    if pre_valid_loss > valid_loss:
                        pre_valid_loss = valid_loss
                        print(f'train loss: {train_loss} / valid loss: {valid_loss} -------------------- epoch: {epoch} iteration: {iteration} ==> save')
                        model.save_pretrained(f'./checkpoints/{file_name}_best1')
                    else:
                        print(f'train loss: {train_loss} / valid loss: {valid_loss} -------------------- epoch: {epoch} iteration: {iteration}')
                    
                    if iteration % (5 * int(len(train_loader) / 10)) == 0:
                        print("scheduler!")
                        scheduler.step(valid_loss)

                    # wandb.log(
                    #     {
                    #         "train_loss": train_loss,
                    #         'valid_loss': valid_loss,
                    #     }
                    #     )

                    train_loss = 0
                    valid_loss = 0
                    model.train()

    model.save_pretrained(f"checkpoints/{file_name}_best0")


    end_time = datetime.now()
    running_time = end_time - start_time
    print(f"END!! {end_time.strftime('%Y_%m_%d / %H_%M')}")
    print(f"RUNNING TIME: {str(running_time)[:7]}")
    print(f'============================================================\n\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='skt/kobert-base-v1')
    parser.add_argument('--trainset', type=str, default='train_1363581')
    parser.add_argument('--validset', type=str, default='valid_170448')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--lang', type=str, default="ko")
    parser.add_argument('--description', type=str, default='')

    args = parser.parse_args()

    main(args)



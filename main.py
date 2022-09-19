import os
import argparse
from datetime import datetime

from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments, Trainer

from modeling import BiEncoder, PolyEncoder
from data_loader import DataLoader
from utils import seed_everything, empty_cuda_cache, pickling

# os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(args):
    print(f'============================================================')
    start_time = datetime.now()
    file_name = f"{args.model}{start_time.strftime('%Y%m%d_%H%M')[2:]}_bs{args.batch * args.accumulation}_ep{args.epoch}_best{args.best}"
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
    print(f'best: {args.best}')
    print(f'description: {args.description}\n')


    seed_everything(args.seed)


    train_context, train_candidate = pickling(f'./data/pickles/{args.trainset}.pickle', 'load')
    valid_context, valid_candidate = pickling(f'./data/pickles/{args.validset}.pickle', 'load')
    print(f"train: {len(train_context)}")
    print(f"valid: {len(valid_context)}")
    print(train_context[0:5])
    print(train_candidate[0:5])
    print()


    tokenizer = KoBERTTokenizer.from_pretrained(args.path)
    if args.model == 'bi':
        model = BiEncoder.from_pretrained(args.path)
    elif args.model == 'poly':
        model = PolyEncoder.from_pretrained(args.path, m=args.m)
    model.to('cuda')


    train_loader = DataLoader(train_context, train_candidate, tokenizer, shuffle = True, seed = args.seed)
    valid_loader = DataLoader(valid_context, valid_candidate, tokenizer, shuffle = True, seed = args.seed)
    

    arguments = TrainingArguments(
        output_dir = 'checkpoints/checkpoints_step',
        do_train = True,
        do_eval = True,

        num_train_epochs = args.epoch,
        learning_rate = args.lr,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size = args.batch,
        gradient_accumulation_steps = args.accumulation,
        dataloader_num_workers=0,

        # warmup_steps = 100,

        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 30,

        evaluation_strategy = "steps",
        eval_steps = 500,

        load_best_model_at_end=args.best,
        
        report_to = 'none',

        fp16=True,
        )

    trainer = Trainer(
        model,
        arguments,
        train_dataset=train_loader,
        eval_dataset=valid_loader
        )


    empty_cuda_cache()
    trainer.train(resume_from_checkpoint = None)
    model.save_pretrained(f"checkpoints/{file_name}")


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
    parser.add_argument('--best', type=int, default=0)
    parser.add_argument('--description', type=str, default='')

    args = parser.parse_args()

    main(args)



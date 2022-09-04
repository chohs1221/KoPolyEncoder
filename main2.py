import os
import argparse
from datetime import datetime

from torch.optim import Adam, lr_scheduler

from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments, Trainer

from modeling import BiEncoder
from parsing import pickling
from data_loader import DataLoader
from utils import seed_everything, empty_cuda_cache

# os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(args):
    print(f'============================================================')
    start_time = datetime.now()
    print(f"START!! {start_time.strftime('%Y_%m_%d / %H_%M')}")
    print(f'seed: {args.seed}')
    print(f'epoch: {args.epoch}')
    print(f'learning rate: {args.lr}')
    print(f'batch size: {args.batch}')
    print(f'accumulate: {args.accumulation}')
    print(f'BEST: {args.best}\n')


    seed_everything(args.seed)


    # train_context, train_candidate = pickling('./data/pickles/train.pickle', 'load')
    # valid_context, valid_candidate = pickling('./data/pickles/valid.pickle', 'load')
    train_context, train_candidate = pickling('./data/pickles/train2.pickle', 'load')
    valid_context, valid_candidate = pickling('./data/pickles/valid2.pickle', 'load')
    print(train_context[100:105])
    print(train_candidate[100:105])
    print()


    # model = BiEncoder.from_pretrained('./checkpoint/firstmodel_ep5')
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    model = BiEncoder.from_pretrained('skt/kobert-base-v1')
    model.to('cuda')


    train_loader = DataLoader(train_context, train_candidate, tokenizer)
    valid_loader = DataLoader(valid_context, valid_candidate, tokenizer)

    # C:\Users\HSC\Documents\VS_workspace\pytorch17_cuda11\lib\site-packages\transformers\trainer.py", line 1810
    # optimizer = Adam(model.parameters(), lr=5e-5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.4)
    arguments = TrainingArguments(
        output_dir = 'checkpoints',
        do_train = True,
        do_eval = True,

        num_train_epochs = args.epoch,
        learning_rate = args.lr,

        warmup_steps = 100,

        save_strategy = "epoch",
        save_total_limit = 10,
        evaluation_strategy = "epoch",
        load_best_model_at_end=args.best,
        
        report_to = 'none',

        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size = args.batch,
        gradient_accumulation_steps = args.accumulation,
        dataloader_num_workers=0,
        fp16=True,
        )

    trainer = Trainer(
        model,
        arguments,
        # optimizers = (optimizer, scheduler),
        train_dataset=train_loader,
        eval_dataset=valid_loader
        )


    empty_cuda_cache()
    trainer.train(resume_from_checkpoint = None)
    model.save_pretrained(f"checkpoints/{start_time.strftime('%Y_%m_%d_%H_%M')}_ep{args.epoch}_bs{args.batch}_ep{args.epoch}")

    end_time = datetime.now()
    running_time = end_time - start_time
    print(f"END!! {end_time.strftime('%Y_%m_%d / %H_%M')}")
    print(f"RUNNING TIME: {str(running_time)[:7]}")
    print(f'============================================================\n\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--best', type=bool, default=False)

    args = parser.parse_args()
    main(args)



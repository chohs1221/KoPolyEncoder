import argparse
import time
from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments, Trainer

from modeling import BiEncoder
from parsing import pickling
from data_loader import DataLoader
from utils import seed_everything, empty_cuda_cache


def main(args):
    # Parse Argument
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)
    print(f'============================================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}\n')


    seed_everything(SEED)

    train_context, train_candidate = pickling('./data/pickle/train.pickle', 'load')
    valid_context, valid_candidate = pickling('./data/pickle/valid.pickle', 'load')
    print(train_context[:5])
    print(train_candidate[:5])


    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    # model = BiEncoder.from_pretrained('./checkpoint/firstmodel_ep5')
    model = BiEncoder.from_pretrained('skt/kobert-base-v1')
    model.to('cuda')


    train_loader = DataLoader(train_context, train_candidate, tokenizer)
    valid_loader = DataLoader(valid_context, valid_candidate, tokenizer)


    arguments = TrainingArguments(
        output_dir='checkpoints',
        do_train=True,
        do_eval=True,

        num_train_epochs=EPOCH,
        learning_rate = LR,

        weight_decay  = 0.01,
        warmup_steps = 100,

        save_strategy="epoch",
        save_total_limit=10,
        evaluation_strategy="epoch",
        load_best_model_at_end=BEST,
        
        report_to = 'none',

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        fp16=True,

    )

    trainer = Trainer(
        model,
        arguments,
        train_dataset=train_loader,
        eval_dataset=valid_loader
    )


    empty_cuda_cache()
    trainer.train()
    model.save_pretrained(f"checkpoints/firstmodel_ep{EPOCH}")


    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    print(f'============================================================\n\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=1)
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--batch', default=256)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--best', default=False)

    args = parser.parse_args()
    main(args)



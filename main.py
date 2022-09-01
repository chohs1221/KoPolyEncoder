from kobert_tokenizer import KoBERTTokenizer
from transformers import TrainingArguments, Trainer

from modeling import BiEncoder
from parsing import pickling
from data_loader import DataLoader
from utils import seed_everything, empty_cuda_cache


seed_everything(42)

train_context, train_candidate = pickling('./data/pickle/train.pickle', 'load')
valid_context, valid_candidate = pickling('./data/pickle/valid.pickle', 'load')
print(train_context[:5])
print(train_candidate[:5])


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BiEncoder.from_pretrained('skt/kobert-base-v1')
model.to('cuda');


train_loader = DataLoader(train_context, train_candidate, tokenizer)
valid_loader = DataLoader(valid_context, valid_candidate, tokenizer)


arguments = TrainingArguments(
    output_dir='checkpoints',
    do_train=True,
    do_eval=True,

    num_train_epochs=5,
    learning_rate = 5e-5,

    weight_decay = 0.4,
    warmup_steps  = 100,

    save_strategy="epoch",
    save_total_limit=10,
    evaluation_strategy="epoch",
    # load_best_model_at_end=True,
    
    report_to = 'none',

    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
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
model.save_pretrained(f"checkpoints/firstmodel_ep5")




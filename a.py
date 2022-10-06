from tqdm import tqdm
from utils import pickling
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ctx, cand = pickling('./data/pickles/ubuntu2_train_1000000.pickle', 'load')

lenght = []
for i in tqdm(range(len(ctx))):
    tok = tokenizer.encode(ctx[i])

    lenght.append(len(tok))

print(sum(lenght) / len(lenght))
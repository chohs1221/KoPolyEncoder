import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from kobert_tokenizer import KoBERTTokenizer

from modeling import BiEncoder, PolyEncoder
from parsing import pickling


def load_tokenizer_model(model_path, m = 0, device = 'cuda'):
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    if m == 0:
        print('Load BiEncoder')
        model = BiEncoder.from_pretrained(model_path).to(device)
    else:
        print('Load PolyEncoder')
        model = PolyEncoder.from_pretrained(model_path, m).to(device)
    
    return tokenizer, model


def candidates_designated(file_dir, model_path, args, device= 'cuda'):
    with open(file_dir, 'r', encoding= 'utf-8') as f:
        candidate_text = [line.strip() for line in f.readlines() if len(line) > 1]
        
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(model_path, args.m, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/{args.path}_designated{len(candidate_text)}.pickle', act= 'load')
    except:
        candidate_input = tokenizer(candidate_text, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
        with torch.no_grad():
            candidate_embeddings = model(**candidate_input, training = False)[:, 0, :]
        
        pickling(f'./data/pickles/{args.path}_designated{len(candidate_text)}.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


def candidates_incorpus(file_dir, model_path, args, batch_size = 256, device = 'cuda'):
    candidate_text0, candidate_text1 = pickling(file_dir, act= 'load')
    candidate_text = candidate_text0 + candidate_text1
    candidate_text = candidate_text[-100000:]
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(model_path, args.m, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/{args.path}.pickle', act= 'load')

    except:
        print('No pickle file exists!!')
        batch = []
        for i in tqdm(range(0, len(candidate_text)-batch_size, batch_size)):
            batch.append(tokenizer(candidate_text[i: i+batch_size], padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device))

        candidate_embeddings = []
        for candidate_input in tqdm(batch):
            with torch.no_grad():
                candidate_embedding = model(**candidate_input, training = False)[:, 0, :]
                candidate_embeddings.append(candidate_embedding)
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)

        pickling(f'./data/pickles/{args.path}.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='2022_09_06_01_19_bs2_ep10')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--incorpus', type=int, default=0)
    args = parser.parse_args()
    print(args)


    device = 'cuda'
    model_path =  './checkpoints/' + args.path
    if args.incorpus:
        tokenizer, model, candidate_text, candidate_embeddings = candidates_incorpus('./data/pickles/test_81068.pickle', model_path, args, batch_size = 256, device=device)
    else:
        tokenizer, model, candidate_text, candidate_embeddings = candidates_designated('./data/responses.txt', model_path, args, device=device)


    while True:
        prompt = input("user >> ")
        if prompt == 'bye' or prompt == 'ㅂㅂ':
            print("{0:>50}\n".format("안녕히 가세요! << bot"))
            break
        print()
        
        try:
            context_input = tokenizer(prompt, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
        except:
            continue
        
        with torch.no_grad():
            if args.model == 'bi':
                context_embedding = model(**context_input, training = False)[:, 0, :]       # (1, hidden state)

                dot_product = torch.matmul(context_embedding, candidate_embeddings.t())     # (1, hidden state) @ (candidate size, hidden state).t() = (1, candidate size)
                dot_product = dot_product[0, :]

            elif args.model == 'poly':
                context_embedding = model(**context_input, training = False, candidate_output = candidate_embeddings)[:, 0, :]  # (candidate size, hidden state)

                dot_product = torch.sum(context_embedding * candidate_embeddings, dim = 1)  # (candidate size)

        sorted_dot_product, indices = torch.sort(F.softmax(dot_product, -1), dim = -1, descending = True)

        print("{0:>50}".format(f"{candidate_text[indices[0]]} << bot ({sorted_dot_product[0] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[1]]} << bot ({sorted_dot_product[1] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[2]]} << bot ({sorted_dot_product[2] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[3]]} << bot ({sorted_dot_product[3] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[4]]} << bot ({sorted_dot_product[4] * 100:.2f}%)"))
        print(list(map(lambda x: round(x*100, 2), sorted_dot_product[:10].tolist())), end='\n\n')


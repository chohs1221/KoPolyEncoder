import argparse
from tqdm import tqdm

import torch

from kobert_tokenizer import KoBERTTokenizer

from modeling import BiEncoder, PolyEncoder
from parsing import pickling


def load_tokenizer_model(model_path, m = 360, device = 'cuda'):
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    if args.model == 'bi':
        model = BiEncoder.from_pretrained(model_path).to(device)
    elif args.model == 'poly':
        model = PolyEncoder.from_pretrained(model_path, m).to(device)
    
    return tokenizer, model


def candidates_designated(file_dir, model_path, args, device= 'cuda'):
    with open(file_dir, 'r', encoding= 'utf-8') as f:
        candidate_text = [line.strip() for line in f.readlines() if len(line) > 1]
        
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(model_path, args.m, device)
    model.eval()

    candidate_input = tokenizer(candidate_text, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
    with torch.no_grad():
        candidate_embeddings = model(**candidate_input, training = False)[:, 0, :]

    return tokenizer, model, candidate_text, candidate_embeddings


def candidates_incorpus(file_dir, model_path, args, batch_size = 256, device = 'cuda'):
    candidate_text0, candidate_text1 = pickling(file_dir, act= 'load')
    candidate_text = candidate_text1
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(model_path, args.m, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/{args.path}.pickle', act= 'load')

    except:
        print('No pickle file exists!!')
        candidate_inputs = []
        for i in tqdm(range(0, len(candidate_text)-batch_size, batch_size)):
            candidate_inputs.append(tokenizer(candidate_text[i: i+batch_size], padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device))

        candidate_embeddings = []
        for candidate_input in tqdm(candidate_inputs):
            with torch.no_grad():
                candidate_embedding = model(**candidate_input, training = False)[:, 0, :]
                candidate_embeddings.append(candidate_embedding)
                candidate_embedding.to('cpu')
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)

        pickling(f'./data/pickles/{args.path}.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='2022_09_06_01_19_bs2_ep10')
    parser.add_argument('--m', type=int, default=360)
    parser.add_argument('--incorpus', type=int, default=0)
    args = parser.parse_args()
    print(args)


    device = 'cuda'
    model_path =  './checkpoints/' + args.path
    if args.incorpus:
        tokenizer, model, candidate_text, candidate_embeddings = candidates_incorpus('./data/pickles/valid.pickle', model_path, args, batch_size = 256, device=device)
    else:
        tokenizer, model, candidate_text, candidate_embeddings = candidates_designated('./data/responses.txt', model_path, args, device=device)


    while True:
        prompt = input("user >> ")
        if prompt == 'bye' or prompt == 'ㅂㅂ':
            print("{0:>50}\n".format("잘가! << bot"))
            break
        print()
        
        context_input = tokenizer(prompt, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
        
        with torch.no_grad():
            if args.model == 'bi':
                context_embedding = model(**context_input, training = False)[:, 0, :]       # (1, hidden state)

                dot_product = torch.matmul(context_embedding, candidate_embeddings.t())     # (1, hidden state) @ (candidate size, hidden state).t() = (1, candidate size)

            elif args.model == 'poly':
                context_embedding = model(**context_input, training = False, candidate_output = candidate_embeddings)[:, 0, :]  # (candidate size, hidden state)

                dot_product = torch.sum(context_embedding * candidate_embeddings, dim = 1)  # (candidate size)

        best_idx = torch.argmax(dot_product).item()

        print("{0:>50}\n".format(candidate_text[best_idx] + " << bot"))


import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from parsing import pickling
from utils import load_tokenizer_model


def get_candidates_dale(file_dir, model_path, args, device= 'cuda'):
    with open(file_dir, 'r', encoding= 'utf-8') as f:
        candidate_text = [line.strip() for line in f.readlines() if len(line) > 1]
    print(len(candidate_text))
    candidate_text = list(dict.fromkeys(candidate_text))
    print(len(candidate_text))
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/candidate/{args.path}_{len(candidate_text)}_dale.pickle', act= 'load')
        print('Pickle file exists!!')
    except:
        print('No pickle file exists!!')
        candidate_input = tokenizer(candidate_text, padding='max_length', max_length=args.max_length, truncation=True, return_tensors = 'pt').to(device)
        with torch.no_grad():
            candidate_embeddings = model.encode(**candidate_input)[:, 0, :]
        
        pickling(f'./data/pickles/candidate/{args.path}_{len(candidate_text)}_dale.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


def get_candidates_corpus(file_dir, model_path, args, batch_size = 256, device = 'cuda'):
    _, candidate_text = pickling(file_dir, act= 'load')
    print(len(candidate_text))
    candidate_text = list(dict.fromkeys(candidate_text))
    print(len(candidate_text))
    candidate_text = candidate_text[-100000:]
    print(f'{len(candidate_text)} candidates found!!')

    tokenizer, model = load_tokenizer_model(args.model, model_path, args.m, args.lang, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/candidate/{args.path}_{len(candidate_text)}_corpus.pickle', act= 'load')
        print('Pickle file exists!!')
    except:
        print('No pickle file exists!!')
        batch = []
        for i in tqdm(range(0, len(candidate_text)-batch_size, batch_size)):
            batch.append(tokenizer(candidate_text[i: i+batch_size], padding='max_length', max_length=args.max_length, truncation=True, return_tensors = 'pt').to(device))

        candidate_embeddings = torch.empty(len(batch) * batch_size, 768).to(device)
        for i, candidate_input in tqdm(enumerate(batch), total = len(batch)):
            with torch.no_grad():
                candidate_embeddings[i*batch_size:(i+1)*batch_size] = model.encode(**candidate_input)[:, 0, :]

        pickling(f'./data/pickles/candidate/{args.path}_{len(candidate_text)}_corpus.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi')
    parser.add_argument('--path', type=str, default='poly221129_1703_bs128_ep5_data52812_ko_best1')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--lang', type=str, default="ko")
    parser.add_argument('--cand', type=str, default="ko_train_1363581")
    parser.add_argument('--corpus', type=int, default=0)
    args = parser.parse_args()
    print(args)


    device = 'cuda'
    model_path =  './checkpoints/' + args.path
    if args.corpus:
        tokenizer, model, candidate_text, candidate_embeddings = get_candidates_corpus(f'./data/pickles/data/{args.cand}.pickle', model_path, args, batch_size = 512, device=device)
    else:
        tokenizer, model, candidate_text, candidate_embeddings = get_candidates_dale(f'./data/responses_{args.lang}.txt', model_path, args, device=device)


    while True:
        prompt = input("user >> ")
        if prompt == 'qq' or prompt == 'ㅂㅂ':
            print("{0:>50}\n".format("안녕히 가세요! << bot"))
            break
        print()
        
        try:
            context_input = tokenizer(prompt, padding='max_length', max_length=args.max_length, truncation=True, return_tensors = 'pt').to(device)
        except:
            continue
        
        with torch.no_grad():
            if args.model == 'bi':
                context_embedding = model.encode(**context_input)[:, 0, :]                      # (1, hidden state)

                dot_product = torch.matmul(context_embedding, candidate_embeddings.t())[0]      # (1, hidden state) @ (candidate size, hidden state).t() = (1, candidate size)

            elif args.model == 'poly':
                context_embedding = model.context_encode(**context_input, candidate_output = candidate_embeddings)[:, 0, :]  # (candidate size, hidden state)

                dot_product = torch.sum(context_embedding * candidate_embeddings, dim = -1)      # (candidate size)

        sorted_dot_product, indices = torch.sort(F.softmax(dot_product, -1), dim = -1, descending = True)

        assert round(torch.sum(sorted_dot_product).item()) == 1

        print("{0:>50}".format(f"{candidate_text[indices[0]]} << bot ({sorted_dot_product[0] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[1]]} << bot ({sorted_dot_product[1] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[2]]} << bot ({sorted_dot_product[2] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[3]]} << bot ({sorted_dot_product[3] * 100:.2f}%)"))
        print("{0:>50}".format(f"{candidate_text[indices[4]]} << bot ({sorted_dot_product[4] * 100:.2f}%)"))
        print(list(map(lambda x: round(x*100, 2), sorted_dot_product[:10].tolist())), end='\n\n')


from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModel
# from kobert_tokenizer import KoBERTTokenizer

from modeling import BiEncoder
from parsing import pickling


def load_tokenizer_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')
    model = AutoModel.from_pretrained(model_path).to(device)
    
    return tokenizer, model


def candidates_designated(file_dir, model_path, device= 'cuda'):
    with open(file_dir, 'r', encoding= 'utf-8') as f:
        candidate_text = []
        for line in f.readlines():
            candidate_text.append(line.strip())

    tokenizer, model = load_tokenizer_model(model_path, device)
    model.eval()

    candidate_input = tokenizer(candidate_text, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
    candidate_embeddings = model(**candidate_input)[0][:, 0, :]

    return tokenizer, model, candidate_text, candidate_embeddings


def candidates_incorpus(file_dir, model_path, batch_size = 256, device = 'cuda'):
    candidate_text0, candidate_text1 = pickling(file_dir, act= 'load')
    candidate_text = candidate_text0 + candidate_text1
    print(f'{len(candidate_text)} candidates found')

    tokenizer, model = load_tokenizer_model(model_path, device)
    model.eval()

    try:
        candidate_embeddings = pickling(f'./data/pickles/corpus_embeddings.pickle', act= 'load')
    except:
        print('No pickle file exists')
        candidate_inputs = []
        for i in tqdm(range(0, len(candidate_text)-batch_size, batch_size)):
            candidate_inputs.append(tokenizer(candidate_text[i: i+batch_size], padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device))

        candidate_embeddings = []
        for candidate_input in tqdm(candidate_inputs):
            with torch.no_grad():
                candidate_embedding = model(**candidate_input)[0][:, 0, :]
                candidate_embeddings.append(candidate_embedding)
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
        pickling('./data/pickles/corpus_embeddings.pickle', act='save', data=candidate_embeddings)

    return tokenizer, model, candidate_text, candidate_embeddings


if __name__ == '__main__':
    device = 'cuda'
    # model_path = 'skt/kobert-base-v1'
    model_path = 'BM-K/KoSimCSE-roberta-multitask'

    tokenizer, model, candidate_text, candidate_embeddings = candidates_incorpus('./data/pickles/valid.pickle', model_path, batch_size = 256, device=device)
    # tokenizer, model, candidate_text, candidate_embeddings = candidates_designated('./data/responses.txt', model_path, device=device)

    while True:
        prompt = input("user >> ")
        if prompt == 'bye':
            break
        
        context_input = tokenizer(prompt, padding='max_length', max_length=50, truncation=True, return_tensors = 'pt').to(device)
        context_embedding = model(**context_input)[0][:, 0, :]

        dot_product = torch.matmul(context_embedding, candidate_embeddings.t())
        
        best_idx = torch.argmax(dot_product).item()

        print(f"bot >>  {candidate_text[best_idx]}")

import os
import random
import pickle
import gc

import numpy as np
import torch

from transformers import BertTokenizer

from kobert_tokenizer import KoBERTTokenizer

from modeling import BiEncoder, PolyEncoder, CrossEncoder


def seed_everything(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def pickling(file_name, act, data = None):
    if act == 'save':
        with open(file_name, 'wb') as fw:
            pickle.dump(data, fw)
    
    elif act == 'load':
        with open(file_name, 'rb') as fr:
            data = pickle.load(fr)
        return data


def load_tokenizer_model(model, model_path, m = 0, lang = "ko", device = 'cuda'):
    if lang == "ko":
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    elif lang == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if model == 'bi':
        print('Load BiEncoder')
        model = BiEncoder.from_pretrained(model_path).to(device)
    elif model == 'poly':
        print('Load PolyEncoder')
        model = PolyEncoder.from_pretrained(model_path, m).to(device)
    elif model == 'cross':
        print('Load CrossEncoder')
        model = CrossEncoder.from_pretrained(model_path).to(device)
    
    return tokenizer, model
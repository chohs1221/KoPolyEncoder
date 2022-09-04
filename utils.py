import os
import random
import pickle
import gc

import numpy as np
import torch


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
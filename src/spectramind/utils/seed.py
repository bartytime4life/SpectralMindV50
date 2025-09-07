import random, numpy as np, torch

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

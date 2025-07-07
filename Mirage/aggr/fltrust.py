import torch
import torch.nn as nn
from torch import linalg as LA

def modelsd2tensor(model_dict):
    ravel_list = []
    for layer_name, parms in model_dict.items():
        ravel_list.append(torch.ravel(parms))
    ravel_list = torch.cat(ravel_list, 0)
    return torch.unsqueeze(ravel_list, 0)


def relu_cos(a, b):
    # a, b as tensors 
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos = cos_sim(a, b)
    
    '''relu'''
    if cos < 0:
        cos = 0
    return cos


def norm_clip(a, b):
    # a, b as tensors
    res = LA.norm(a)/(LA.norm(b)+1e-9)
    return res


def cosScoreAndClipValue(model_dict1, model_dict2):
    t1 = modelsd2tensor(model_dict1)
    t2 = modelsd2tensor(model_dict2)
    return relu_cos(t1, t2), norm_clip(t1, t2)
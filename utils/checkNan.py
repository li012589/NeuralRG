import torch

def checkNan(x):
    assert torch.isnan(x).sum().item() == 0
    return x
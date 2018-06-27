import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class debugRealNVP(nn.Module):
    def __init__(self):
        super(debugRealNVP,self).__init__()
    def inverse(self,x):
        return x, x.new_ones(x.shape[0])
    def forward(self,x):
        return x, x.new_ones(x.shape[0])


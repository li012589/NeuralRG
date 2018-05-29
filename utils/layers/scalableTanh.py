import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScalableTanh(nn.Module):
    def __init__(self,input_size):
        super(ScalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.zeros(input_size))
    def forward(self,x):
        return self.scale * F.tanh(x)


import math
import torch
import torch.nn as nn

class ScalableTanh(nn.Module):
    def __init__(self,input_size):
        super(ScalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.zeros(input_size),requires_grad=True)
    def forward(self,x):
        return self.scale * torch.tanh(x)


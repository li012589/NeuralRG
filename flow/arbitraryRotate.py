
import torch
from torch import nn
import numpy as np
import scipy

from .flow import Flow

class ArbitraryRotate(Flow):
    def __init__(self, c,prior = None, name = "ArbitraryRotate"):
        super(ArbitraryRotate,self).__init__(prior,name)
        self.w = nn.Parameter(torch.randn(c,c).to(torch.float32))

    def inverse(self,y):
        inverseLogjac = torch.log(torch.abs(torch.det(self.w)))*y.shape[-1]*y.shape[-2]*torch.ones(y.shape[0])
        print(self.w)

        y = torch.matmul(y.permute([0,2,3,1]),self.w.reshape(1,1,*self.w.shape)).permute(0,3,1,2)
        return y,inverseLogjac

    def forward(self,z):
        w_ = torch.inverse(self.w)
        forwardLogjac = torch.log(torch.abs(torch.det(w_)))*z.shape[-1]*z.shape[-2]*torch.ones(z.shape[0])
        print(w_)

        z = torch.matmul(z.permute([0,2,3,1]),w_.reshape(1,1,*w_.shape)).permute(0,3,1,2)
        return z,forwardLogjac

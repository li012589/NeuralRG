import torch
from torch import nn

from .flow import Flow
from utils import checkNan

class Rotate(Flow):
    def __init__(self, prior = None, name = "rotate"):
        super(Rotate,self).__init__(prior,name)
        self.theta = nn.Parameter(torch.randn(1))

    def inverse(self,y):
        w = torch.tensor([[torch.cos(self.theta),torch.sin(self.theta)],[-torch.sin(self.theta),torch.cos(self.theta)]]).to(self.theta).reshape(1,2,2)
        inverseLogjac = torch.zeros(y.shape[0]).to(self.theta)
        y = torch.matmul(w,y)
        return y,inverseLogjac

    def forward(self,z):
        w = torch.tensor([[torch.cos(self.theta),-torch.sin(self.theta)],[torch.sin(self.theta),torch.cos(self.theta)]]).to(self.theta).reshape(1,2,2)
        forwardLogjac = torch.zeros(z.shape[0]).to(self.theta)
        z = torch.matmul(w,z)
        return z,forwardLogjac

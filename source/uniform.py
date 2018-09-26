import torch
import numpy as np

from .source import Source

class Uniform(Source):
    def __init__(self, nvars, a, b, name="gaussian", requiresGrad = False):
        super(Uniform,self).__init__(nvars,name)
        assert b>a
        self.a = torch.nn.Parameter(torch.tensor([a],dtype=torch.float32),requires_grad=requiresGrad)
        self.b = torch.nn.Parameter(torch.tensor([b],dtype=torch.float32),requires_grad=requiresGrad)

    def sample(self, batchSize):
        size = [batchSize] + self.nvars
        return (torch.rand(size)*(self.b-self.a)+self.a).to(self.a)

    def energy(self, z):
        mask = z<self.b
        maskp = z>self.a
        mask = (mask*maskp).to(self.a)
        return -mask*torch.ones_like(z)*(torch.log((self.b-self.a)**-1))

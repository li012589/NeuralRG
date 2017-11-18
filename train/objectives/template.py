import torch
from torch.autograd import Variable
import numpy as np

class Target(object):
    '''
    base class for target 
    '''
    def __init__(self,nvars,name = "Target"):
        self.nvars = nvars
        self.name = name

    def __call__(self, x):
        z = Variable(x,requires_grad=True)
        return self.energy(z).data

    def energy(self,z):
        raise NotImplementedError(str(type(self)))

    def backward(self,z):
        z = Variable(z,requires_grad=True)
        out = self.energy(z)
        batchSize = z.size()[0]
        out.backward(torch.ones(batchSize))
        return z.grad.data

    def measure(self, x):
        return (x**2).sum(dim=1).cpu().numpy()
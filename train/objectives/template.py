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
        return self.energy(x)

    def energy(self,z):
        raise NotImplementedError(str(type(self)))

    def backward(self,z):
        z = Variable(z,requires_grad=True)
        out = self.energy(z)
        batchSize = z.size()[0]
        if isinstance(z.data,torch.DoubleTensor):
            out.backward(torch.ones(batchSize).double())
        else:
            out.backward(torch.ones(batchSize))
        return z.grad.data

    def measure(self, x):
        return (x.data**2).sum(dim=1).cpu().numpy()

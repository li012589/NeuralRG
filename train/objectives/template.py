import torch
from torch.autograd import Variable
import numpy as np

class Target(object):
    '''
    base class for target 
    '''
    def __init__(self,nvars,name = "Target", beta=1.0):
        self.nvars = nvars
        self.name = name
        self.beta = beta

    def __call__(self, x):
        return self.energy(x)

    def energy(self,z):
        raise NotImplementedError(str(type(self)))

    def backward(self,z):
        z = Variable(z.data,requires_grad=True)
        out = self.energy(z)
        batchSize = z.size()[0]
        if isinstance(z.data,torch.DoubleTensor):
            out.backward(torch.ones(batchSize).double())
        else:
            out.backward(torch.ones(batchSize))
        return Variable(z.grad.data,requires_grad = True)

    def measure(self, x):
        return (x.data**2).sum(dim=1).cpu().numpy()

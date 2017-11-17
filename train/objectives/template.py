import torch 
import numpy as np

class Target(object):
    '''
    base class for target 
    '''
    def __init__(self,nvars,name = "Target"):
        self.nvars = nvars
        self.name = name

    def __call__(self, x):
        raise NotImplementedError(str(type(self)))

    def measure(self, x):
        return (x**2).sum(dim=1).cpu().numpy()
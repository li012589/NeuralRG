import numpy as np
import torch
from torch import nn

class Source(nn.Module):

    def __init__(self, nvars,name = "Flow"):
        super(Source, self).__init__()
        self.name = name
        self.nvars = nvars

    def __call__(self,*args,**kargs):
        return self.sample(*args,**kargs)

    def sample(self,batchSize):
        raise NotImplementedError(str(type(self)))

    def logProbability(self,x):
        return -self.energy(x)

    def energy(self,x):
        raise NotImplementedError(str(type(self)))

    def save(self,saveDict):
        saveDict[self.name] = self.state_dict()
        return saveDict

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict
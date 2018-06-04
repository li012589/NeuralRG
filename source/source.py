import numpy as np
import torch
from torch import nn
from utils import HMC, Metropolis

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

    def save(self):
        return self.state_dict()

    def load(self,saveDict):
        self.load_state_dict(saveDict)
        return saveDict

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)


    def _sampleWithHMC(self,batchSize,thermalSteps = 50, interSteps = 5, epsilon=0.1):
        inital = torch.randn([batchSize]+self.nvars,requires_grad=True)
        inital = HMC(self.energy,inital,thermalSteps,interSteps,epsilon)
        return inital.detach()

    def _sampleWithMetropolis(self,batchSize,thermalSteps = 100,tranCore = None):
        inital = torch.randn([batchSize]+self.nvars,requires_grad=True)
        inital = Metropolis(self.energy,inital,thermalSteps,tranCore)
        return inital.detach()
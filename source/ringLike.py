import numpy as np
import torch

from .source import Source

class Ring2d(Source):
    def __init__(self):
        super(Ring2d,self).__init__([2],'Ring2D')

    def sample(self,batchSize,thermalSteps = 100, tranCore=None):
        return self._sampleWithMetropolis(batchSize,thermalSteps,tranCore)
    '''
    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)
    '''

    def energy(self,x):
        return  (torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32#-(torch.sqrt((x**2).sum(dim=-1))-2.0)**2/0.32
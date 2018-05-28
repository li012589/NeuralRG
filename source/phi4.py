import numpy as np
import torch

from .source import Source
from utils import HMC
from utils import roll

class Phi4(Source):
    def __init__(self,l,dims,kappa,lamb,name = None):
        if name is None:
            self.name = "phi4_l"+str(l)+"_d"+str(dims)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            self.name = name
        self.kappa = kappa
        self.lamb = lamb
        self.dims = dims
        nvars = []
        for _ in range(dims):
            nvars += [l]
        super(Phi4,self).__init__(nvars,name)

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        inital = torch.randn([batchSize]+self.nvars,requires_grad=True)
        inital = HMC(self.energy,inital,thermalSteps,interSteps,epsilon)
        return inital.detach()

    def energy(self,x):
        S = 0
        for i in range(self.dims):
            S += x*roll(x,[1],[i+1])
            #S += x*roll(x,[-1],[i+1])
        term1 = x**2
        term2 = (term1-1)**2
        for _ in range(self.dims):
            S = S.sum(-1)
            term1 = term1.sum(-1)
            term2 = term2.sum(-1)
        S *= -2*self.kappa
        term2 *= self.lamb
        S += term1 + term2
        return -S
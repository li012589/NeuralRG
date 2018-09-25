import numpy as np
import torch
from torch import nn

def MetropolisWithAccept(energy,x,length,tranCore = None):
    torch.set_grad_enabled(False)
    shape = [i if no==0 else 1 for no,i in enumerate(x.shape)]
    if tranCore is None:
        def t(x):
            return x + torch.randn_like(x)
        tranCore = t
    E = energy(x)
    for l in range(length):
        xnew = tranCore(x)
        Enew = energy(xnew)

        diff = E-Enew
        accept = (diff.exp()>=diff.uniform_()).to(x)

        E = accept*Enew + (1.-accept)*E
        acceptMask = accept.reshape(shape)
        x = acceptMask*xnew+(1.-acceptMask)*x
    torch.set_grad_enabled(True)

    return x, accept

def Metropolis(*args,**kwargs):
    x, _ = MetropolisWithAccept(*args,**kwargs)
    return x

class MetroplolisSampler(nn.Module):
    def __init__(self,energy,nvars,thermalSteps = 25,tranCore = None):
        super(MetroplolisSampler,self).__init__()
        if tranCore is None:
            def t(x):
                return x + torch.randn_like(x)
            tranCore = t
        self.nvars = nvars
        self.energy = energy
        self.inital = Metropolis(self.energy,torch.randn(nvars),thermalSteps,self.tranCore)

    def step(self):
        self.inital = Metropolis(self.energy,torch.randn(nvars),thermalSteps,self.tranCore)
        return self.inital
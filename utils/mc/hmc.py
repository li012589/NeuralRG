import numpy as np
import torch
from torch import nn
from torch.autograd import grad as torchgrad

torch.manual_seed(42)

def HMC(energy,x,length,steps,epsilon):
    def grad(z):
        return torchgrad(energy(z),z,grad_outputs=torch.ones(z.shape[0]))[0]

    E = energy(x)
    g = grad(x)

    for l in range(length):
        p = x.new_empty(size=x.size()).normal_()
        H = (0.5*p*p).view(p.shape[0], -1).sum(dim=1) + E
        xnew = x
        gnew = g
        for _ in range(steps):
            p = p- epsilon* gnew/2.
            xnew = xnew + epsilon * p
            gnew = grad(xnew)
            p = p- epsilon* gnew/2.
        Enew = energy(xnew)
        Hnew = (0.5*p*p).view(p.shape[0], -1).sum(dim=1) + Enew
        diff = H-Hnew
        accept = (diff.exp() >= diff.uniform_()).to(x)

        E = accept*Enew + (1.-accept)*E
        accept = accept.view(x.shape[0], 1, 1)
        x = accept*xnew + (1.-accept)*x
        g = accept*gnew + (1.-accept)*g

    return x


class HMCsampler(nn.Module):
    def __init__(self,energy,nvars, epsilon=0.01, interSteps=10 , thermalSteps = 10):
        super(HMCsampler,self).__init__()
        self.nvars = nvars
        self.energy = energy
        self.interSteps = interSteps
        self.inital = HMC(self.energy,torch.randn(nvars),thermalSteps,interSteps)

    def step(self):
        return HMC(self.energy,self.inital,1,interSteps,epsilon)
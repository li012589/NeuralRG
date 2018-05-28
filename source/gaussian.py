import torch
import numpy as np

from .source import Source

class Gaussian(Source):
    def __init__(self, nvars, sigma = 1, name="gaussian", requiresGrad = False):
        super(Gaussian,self).__init__(nvars,name)
        self.sigma = torch.nn.Parameter(torch.tensor([sigma],dtype=torch.float32),requires_grad=requiresGrad)

    def sample(self, batchSize):
        size = [batchSize] + self.nvars
        return torch.randn(size).to(self.sigma)

    def energy(self, z):
        return -(-0.5 * (z/self.sigma)**2-0.5*torch.log(2.*np.pi*self.sigma**2)).view(z.shape[0],-1).sum(dim=1)
import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC, HMCSampler, fit

class Buffer(object):
    def __init__(self,maximum,data=None):
        self.data = data
        self.maximum = maximum
    def draw(self,batchSize):
        if batchSize >self.data.shape[0]:
            batchSize = self.data.shape[0]
        perm = np.random.permutation(self.data.shape[0])
        return self.data[perm[:batchSize]]
    def push(self,data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate([self.data,data],axis=0)
        if self.data.shape[0] > self.maximum:
            self._kill()
    def _kill(self):
        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm[:self.maximum]]

def boot():
    pass

def strap():
    pass

def main():
    maximum = 100
    buf = Buffer(maximum)
    buf.push(np.random.randn(200,5))
    print(buf.data.shape)
    print(buf.draw(10).shape)


if __name__ == "__main__":
    main()


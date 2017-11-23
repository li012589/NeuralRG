import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC, HMCSampler, fit

class buffer(object):
    def __init__(maximum):
        pass
    def draw(batchSize):
        pass
    def push(data):
        pass
    def _kill(batchSize):
        pass

def boot():
    pass

def strap():
    pass

def main():
    pass


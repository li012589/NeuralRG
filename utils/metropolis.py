import numpy as np
import torch
from torch.autograd import Variable

def metropolis(e1,e2):
    diff = e1-e2
    return diff.exp()-diff.uniform_()>=0.0
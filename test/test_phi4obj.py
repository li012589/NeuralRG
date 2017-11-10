import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal
#from model import Gaussian,MLP,RealNVP,CNN
from train.objectives import phi4

#from subprocess import Popen, PIPE
#import pytest

def test_phi4():
    z = torch.Tensor([[1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,10]])
    obj = phi4(3,2,1,1)
    e = obj(z).numpy()
    #print(e)
    results = np.array([14097.0,23601.0])
    assert_array_almost_equal(results,e)

if __name__ == "__main__":
    test_phi4()
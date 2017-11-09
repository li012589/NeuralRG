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
    pass

if __name__ == "__main__":
    pass
import os
import sys
sys.path.append(os.getcwd())

from profilehooks import profile

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal
from model import Gaussian,MLP,RealNVP,CNN,GMM,Cauchy

from subprocess import Popen, PIPE
import pytest

noCuda = 0
try:
    p  = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode", "--format=csv,noheader,nounits"], stdout=PIPE)
except OSError:
    noCuda = 1

maxGPU = 0
if noCuda == 0:
    try:
        p = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader,nounits')
        i = p.read().split('\n')
        maxGPU = int(i[-2])+1
    except OSError:
        noCuda = 1

skipIfNoCuda = pytest.mark.skipif(noCuda == 1,reason = "No cuda insatllation, found through nvidia-smi")
skipIfOnlyOneGPU = pytest.mark.skipif(maxGPU < 2,reason = "Only one gpu")

def test_gaussian():
    prior = Gaussian([2,4,4])
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    assert a.shape[2] == 4
    assert a.shape[3] == 4
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)

@skipIfNoCuda
def test_gaussian_cuda():
    prior = Gaussian([2,4,4]).cuda
    prior.cudaNo = 0
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    assert a.shape[2] == 4
    assert a.shape[3] == 4
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)

def test_gaussian_double():
    prior = Gaussian([2,4,4],double = True)
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    assert a.shape[2] == 4
    assert a.shape[3] == 4
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)


def test_GMM():
    prior = GMM([2])
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)

@skipIfNoCuda
def test_GMM_cuda():
    prior = GMM([2]).cuda
    prior.cudaNo = 0
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)

def test_GMM_double():
    prior = GMM([2],double = True)
    a = prior.sample(5)
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    b = prior.logProbability(a)
    assert b.shape[0] == 5
    assert len(b.shape) == 1
    prior(5)

if __name__ == "__main__":
    test_GMM()
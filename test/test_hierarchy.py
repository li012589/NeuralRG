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
from model import RealNVPtemplate,MLP,CNN,RealNVP,Gaussian
from hierarchy import Roll, Wide2bacth, Batch2wide, Placeholder, Mask

from hierarchy import HierarchyBijector

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

skipIfNoCuda = pytest.mark.skipif(noCuda == 1,reason = "NO cuda insatllation, found through nvidia-smi")
skipIfOnlyOneGPU = pytest.mark.skipif(maxGPU < 2,reason = "Only one gpu")

def test_mera_1d():
    mask = Variable(torch.zeros(2).byte())
    masks = [Variable(torch.ByteTensor([1,0,1,0,1,0,1,0])),Variable(torch.ByteTensor([1,0,0,0,1,0,0,0]))]
    masks_ = [Variable(torch.ByteTensor([0,1,0,1,0,1,0,1])),Variable(torch.ByteTensor([0,1,1,1,0,1,1,1]))]

    rollList = [Placeholder(),Roll(1,1),Placeholder(),Roll(1,1),Placeholder(),Roll(1,1)]
    maskList = [Placeholder(2),Placeholder(2),Mask(masks[0],masks_[0]),Mask(masks[0],masks_[0]),Mask(masks[1],masks_[1]),Mask(masks[1],masks_[1])]
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLP(2, Hs) for _ in range(Nlayers)]
    tList = [MLP(2, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8])
    layers = [RealNVP([2],
                      sList,
                      tList,
                      Gaussian([2]),
                      masktypelist) for _ in range(6)]
    model = HierarchyBijector(1,[2 for _ in range(6)],rollList,layers,maskList,None)
    z = prior(4)
    print(z)
    x = model.inference(z,True)
    print(x)
    fLog = model._inferenceLogjac
    print(model._inferenceLogjac)
    zz = model.generate(x,True)
    print(zz)
    bLog = model._generateLogjac
    print(model._generateLogjac)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy())
    assert_array_almost_equal(fLog.numpy(),-bLog.numpy())

def test_mera_2d():
    pass

@skipIfNoCuda
def test_mera_1d_cuda():
    pass

if __name__ == "__main__":
    test_mera_1d()
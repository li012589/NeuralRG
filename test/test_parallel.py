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
from model import Gaussian,MLP,RealNVP,CNN,parallelize

from subprocess import Popen, PIPE
import pytest

noCuda = 0
try:
    p  = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode", "--format=csv,noheader,nounits"], stdout=PIPE)
except OSError:
    noCuda = 1

skipIfNoCuda = pytest.mark.skipif(noCuda == 1,reason = "NO cuda insatllation, found through nvidia-smi")

@skipIfNoCuda
def test_parallel():
    gaussian3d = Gaussian([2,4,4])
    x3d = gaussian3d(1000)
    #z3dp = z3d[:,0,:,:].view(10,-1,4,4)
    #print(z3dp)

    netStructure = [[3,2,1,1],[4,2,1,1],[3,2,1,0],[1,2,1,0]] # [channel, filter_size, stride, padding]

    sList3d = [CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure)]
    tList3d = [CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure)]

    realNVP3d = RealNVP([2,4,4], sList3d, tList3d, gaussian3d)
    #mask3d = realNVP3d.createMask("checkerboard")

    z3d = realNVP3d.generate(x3d,2)
    #print("3d forward:")
    #print(z3d)

    zp3d = realNVP3d.inference(z3d,2)
    #print("Backward")
    #print(zp3d)

    #print("3d logProbability")
    b = (realNVP3d.logProbability(z3d,2))
    a = parallelize(realNVP3d,[0,1,2],"logProbability",x3d,(2,))

    print(a)

    assert_array_almost_equal(b.data.numpy(),a.cpu().data.numpy())

if __name__ == "__main__":
    test_parallel()
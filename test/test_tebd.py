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
from model import RealNVP, TEBD , Gaussian, MLP 

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

def test_invertible():

    #RNVP block
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
                      masktypelist) for _ in range(4)] 
    
    model = TEBD(prior, layers)

    z = model.prior(10)

    print("original")

    x = model.generate(z)
    print("Forward")

    zp = model.inference(x)

    print("Backward")
    assert_array_almost_equal(z.data.numpy(),zp.data.numpy())

    saveDict = model.saveModel({})
    torch.save(saveDict, './saveNet.testSave')

    sListp = [MLP(2, Hs) for _ in range(Nlayers)]
    tListp = [MLP(2, Ht) for _ in range(Nlayers)]
    masktypelistp = ['channel', 'channel'] * (Nlayers//2)
    
    #assamble RNVP blocks into a TEBD layer
    priorp = Gaussian([8])
    layersp = [RealNVP([2], 
                      sListp, 
                      tListp, 
                      Gaussian([2]), 
                       masktypelistp) for _ in range(4)] 
    
    modelp = TEBD(priorp, layersp)
    saveDictp = torch.load('./saveNet.testSave')
    modelp.loadModel(saveDictp)

    xp = modelp.generate(z)

    #modelp = RealNVP([2], sListp, tListp, gaussian)
    #saveDictp = torch.load('./saveNet.testSave')
    #modelp.loadModel(saveDictp)

    #xx = model.generate(z)
    #print("Forward after restore")

    assert_array_almost_equal(xp.data.numpy(),x.data.numpy())

if __name__=='__main__':
    test_invertible()

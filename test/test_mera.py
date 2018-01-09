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
from model import RealNVP, Gaussian, MLP
from hierarchy import MERA, MLPreshape,debugRealNVP, Roll
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

def test_invertible_1d():
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
    model = MERA(1,2,8,layers,prior)
    z = prior(4)
    x = model.generate(z, ifLogjac=True)
    zz = model.inference(x, ifLogjac=True)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy())
    print (model._generateLogjac)
    print (model._inferenceLogjac)
    assert_array_almost_equal(model._generateLogjac.data.numpy(), -model._inferenceLogjac.data.numpy())

    saveDict = model.saveModel({})
    torch.save(saveDict, './saveNet.testSave')

    Nlayersp = 4
    Hsp = 10
    Htp = 10
    sListp = [MLP(2, Hsp) for _ in range(Nlayersp)]
    tListp = [MLP(2, Htp) for _ in range(Nlayersp)]
    masktypelistp = ['channel', 'channel'] * (Nlayersp//2)
    #assamble RNVP blocks into a TEBD layer
    priorp = Gaussian([8])
    layersp = [RealNVP([2],
                      sListp,
                      tListp,
                      Gaussian([2]),
                       masktypelistp) for _ in range(6)]
    modelp = MERA(1,2,8,layersp,priorp)

    saveDictp = torch.load('./saveNet.testSave')
    modelp.loadModel(saveDictp)

    xp = modelp.generate(z)

    assert_array_almost_equal(xp.data.numpy(),x.data.numpy())

def test_translationalinvariance_1d():
    Nlayers = 2 
    Hs = 10
    Ht = 10
    sList = [MLP(2, Hs) for _ in range(Nlayers)]
    tList = [MLP(2, Ht) for _ in range(Nlayers)]
    masktypelist = ['evenodd', 'evenodd'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8])
    layers = [RealNVP([2],
                      sList,
                      tList,
                      Gaussian([2]),
                      masktypelist) for _ in range(6)]
    model = MERA(1,2,8,layers,prior)

    x = model.sample(10)
    xright = Roll(4,1).forward(x)
    xleft = Roll(-4,1).forward(x)

    logp = model.logProbability(x)
    assert_array_almost_equal(logp.data.numpy(),model.logProbability(xleft).data.numpy(), decimal=6)
    assert_array_almost_equal(logp.data.numpy(),model.logProbability(xright).data.numpy(), decimal=6)

def test_translationalinvariance_2d():
    Nlayers = 2 
    Hs = 10
    Ht = 10
    sList = [MLPreshape(4, Hs) for _ in range(Nlayers)]
    tList = [MLPreshape(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['evenodd', 'evenodd'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([4,4])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      None,
                      masktypelist) for _ in range(4)]
    model = MERA(2,[2,2],16,layers,prior)

    x = model.sample(10)
    xright = Roll([2,2],[1,2]).forward(x)
    xleft = Roll([-2,-2],[1,2]).forward(x)

    logp = model.logProbability(x)
    assert_array_almost_equal(logp.data.numpy(),model.logProbability(xleft).data.numpy(), decimal=5)
    assert_array_almost_equal(logp.data.numpy(),model.logProbability(xright).data.numpy(), decimal=5)


def test_invertible_2d():

    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLPreshape(4, Hs) for _ in range(Nlayers)]
    tList = [MLPreshape(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8,8])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(6)]
    #layers = [debugRealNVP() for _ in range(6)]
    model = MERA(2,[2,2],64,layers,prior)
    #z = prior(1)
    z = Variable(torch.from_numpy(np.arange(64)).float().view(1,8,8))
    x = model.generate(z)
    zz = model.inference(x)

    print(zz)
    print(z)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy(),decimal=4) # don't work for decimal >=5, maybe caz by float

    saveDict = model.saveModel({})
    torch.save(saveDict, './saveNet.testSave')

    Nlayersp = 4
    Hsp = 10
    Htp = 10
    sListp = [MLPreshape(4, Hsp) for _ in range(Nlayersp)]
    tListp = [MLPreshape(4, Htp) for _ in range(Nlayersp)]
    masktypelistp = ['channel', 'channel'] * (Nlayersp//2)
    #assamble RNVP blocks into a TEBD layer
    priorp = Gaussian([8,8])
    layersp = [RealNVP([2,2],
                      sListp,
                      tListp,
                      Gaussian([2,2]),
                       masktypelistp) for _ in range(6)]
    modelp = MERA(2,[2,2],64,layersp,priorp)

    saveDictp = torch.load('./saveNet.testSave')
    modelp.loadModel(saveDictp)

    xp = modelp.generate(z)

    assert_array_almost_equal(xp.data.numpy(),x.data.numpy())

@skipIfOnlyOneGPU
def test_invertible_2d_cuda():
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLPreshape(4, Hs) for _ in range(Nlayers)]
    tList = [MLPreshape(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8,8])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(6)]
    #layers = [debugRealNVP() for _ in range(6)]
    model = MERA(2,[2,2],64,layers,prior).cuda()
    #z = prior(1)
    z = Variable(torch.from_numpy(np.arange(64)).float().view(1,8,8)).cuda()
    x = model.generate(z)
    zz = model.inference(x)

    print(zz)
    print(z)

    assert_array_almost_equal(z.data.cpu().numpy(),zz.data.cpu().numpy(),decimal=4) # don't work for decimal >=5, maybe caz by float

    saveDict = model.saveModel({})
    torch.save(saveDict, './saveNet.testSave')

    Nlayersp = 4
    Hsp = 10
    Htp = 10
    sListp = [MLPreshape(4, Hsp) for _ in range(Nlayersp)]
    tListp = [MLPreshape(4, Htp) for _ in range(Nlayersp)]
    masktypelistp = ['channel', 'channel'] * (Nlayersp//2)
    #assamble RNVP blocks into a TEBD layer
    priorp = Gaussian([8,8])
    layersp = [RealNVP([2,2],
                      sListp,
                      tListp,
                      Gaussian([2,2]),
                       masktypelistp) for _ in range(6)]
    modelp = MERA(2,[2,2],64,layersp,priorp)

    saveDictp = torch.load('./saveNet.testSave')
    modelp.loadModel(saveDictp)
    modelp = modelp.cuda(1)

    xp = modelp.generate(z.cuda(1))

    assert_array_almost_equal(xp.data.cpu().numpy(),x.data.cpu().numpy())

def test_invertible_2d_metaDepth3():
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLPreshape(4, Hs) for _ in range(Nlayers)]
    tList = [MLPreshape(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8,8])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(9)]
    #layers = [debugRealNVP() for _ in range(6)]
    model = MERA(2,[2,2],64,layers,prior,metaDepth = 3)
    z = Variable(torch.from_numpy(np.arange(64)).float().view(1,8,8))
    x = model.generate(z)
    zz = model.inference(x)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy(),decimal=4) # don't work for decimal >=5, maybe caz by float

if __name__ == "__main__":
    #test_translationalinvariance_1d()
    #test_invertible_2d_cuda()
    test_invertible_2d_metaDepth3()

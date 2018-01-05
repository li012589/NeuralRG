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
from hierarchy import Roll, Wide2bacth, Batch2wide, Placeholder, Mask, MLP2d

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
    model = HierarchyBijector(1,[2 for _ in range(6)],rollList,layers,maskList,Gaussian([8]))
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

    print(model.sample(10))

    saveDict = model.saveModel({})
    torch.save(saveDict,'./savetest.testSave')

    masksp = [Variable(torch.ByteTensor([1,0,1,0,1,0,1,0])),Variable(torch.ByteTensor([1,0,0,0,1,0,0,0]))]
    masks_p = [Variable(torch.ByteTensor([0,1,0,1,0,1,0,1])),Variable(torch.ByteTensor([0,1,1,1,0,1,1,1]))]

    rollListp = [Placeholder(),Roll(1,1),Placeholder(),Roll(1,1),Placeholder(),Roll(1,1)]
    maskListp = [Placeholder(2),Placeholder(2),Mask(masksp[0],masks_p[0]),Mask(masksp[0],masks_p[0]),Mask(masksp[1],masks_p[1]),Mask(masksp[1],masks_p[1])]
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
    modelp = HierarchyBijector(1,[2 for _ in range(6)],rollListp,layersp,maskListp,Gaussian([8]))

    saveDictp = torch.load('./savetest.testSave')
    modelp.loadModel(saveDictp)

    xp = modelp.inference(z)

    print(xp)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy())
    assert_array_almost_equal(fLog.numpy(),-bLog.numpy())
    assert_array_almost_equal(xp.data.numpy(),x.data.numpy())

def test_mera_2d():
    masks = [Variable(torch.ByteTensor([[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]))]
    masks_ = [Variable(torch.ByteTensor([[0,1,0,1],[1,1,1,1],[0,1,0,1],[1,1,1,1]]))]

    rollList = [Placeholder(),Roll([1,1],[1,2]),Placeholder(),Roll([1,1],[1,2])]
    maskList = [Placeholder(2),Placeholder(2),Mask(masks[0],masks_[0]),Mask(masks[0],masks_[0])]
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLP2d(4, Hs) for _ in range(Nlayers)]
    tList = [MLP2d(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([4,4])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(4)]
    model = HierarchyBijector(2,[[2,2] for _ in range(4)],rollList,layers,maskList,Gaussian([4,4]))
    z = prior(2)
    print(z)
    x = model.inference(z,True)
    print(x)
    fLog = model._inferenceLogjac
    print(model._inferenceLogjac)
    zz = model.generate(x,True)
    print(zz)
    bLog = model._generateLogjac
    print(model._generateLogjac)

    print(model.sample(10))

    saveDict = model.saveModel({})
    torch.save(saveDict,'./savetest.testSave')

    masksp = [Variable(torch.ByteTensor([[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]))]
    masks_p = [Variable(torch.ByteTensor([[0,1,0,1],[1,1,1,1],[0,1,0,1],[1,1,1,1]]))]

    rollListp = [Placeholder(),Roll([1,1],[1,2]),Placeholder(),Roll([1,1],[1,2])]
    maskListp = [Placeholder(2),Placeholder(2),Mask(masksp[0],masks_p[0]),Mask(masksp[0],masks_p[0])]
    Nlayersp = 4
    Hsp = 10
    Htp = 10
    sListp = [MLP2d(4, Hsp) for _ in range(Nlayersp)]
    tListp = [MLP2d(4, Htp) for _ in range(Nlayersp)]
    masktypelistp = ['channel', 'channel'] * (Nlayersp//2)
    #assamble RNVP blocks into a TEBD layer
    priorp = Gaussian([4,4])
    layersp = [RealNVP([2,2],
                      sListp,
                      tListp,
                      Gaussian([2,2]),
                       masktypelistp) for _ in range(4)]
    modelp = HierarchyBijector(2,[[2,2] for _ in range(4)],rollListp,layersp,maskListp,Gaussian([4,4]))

    saveDictp = torch.load('./savetest.testSave')
    modelp.loadModel(saveDictp)

    xp = modelp.inference(z)

    print(xp)

    assert_array_almost_equal(z.data.numpy(),zz.data.numpy())
    assert_array_almost_equal(fLog.numpy(),-bLog.numpy())
    assert_array_almost_equal(xp.data.numpy(),x.data.numpy())

@skipIfNoCuda
def test_mera_2d_cuda():
    masks = [Variable(torch.ByteTensor([[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]))]
    masks_ = [Variable(torch.ByteTensor([[0,1,0,1],[1,1,1,1],[0,1,0,1],[1,1,1,1]]))]

    rollList = [Placeholder(),Roll([1,1],[1,2]),Placeholder(),Roll([1,1],[1,2])]
    maskList = [Placeholder(2),Placeholder(2),Mask(masks[0],masks_[0]),Mask(masks[0],masks_[0])]
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLP2d(4, Hs) for _ in range(Nlayers)]
    tList = [MLP2d(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([4,4])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(4)]
    model = HierarchyBijector(2,[[2,2] for _ in range(4)],rollList,layers,maskList,None).cuda()
    z = prior(2).cuda()
    print(z)
    x = model.inference(z,True)
    print(x)
    fLog = model._inferenceLogjac
    print(model._inferenceLogjac)
    zz = model.generate(x,True)
    print(zz)
    bLog = model._generateLogjac
    print(model._generateLogjac)

    assert_array_almost_equal(z.data.cpu().numpy(),zz.data.cpu().numpy())
    assert_array_almost_equal(fLog.cpu().numpy(),-bLog.cpu().numpy())

@skipIfOnlyOneGPU
def test_mera_2d_cudaNotOne():
    masks = [Variable(torch.ByteTensor([[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]))]
    masks_ = [Variable(torch.ByteTensor([[0,1,0,1],[1,1,1,1],[0,1,0,1],[1,1,1,1]]))]

    rollList = [Placeholder(),Roll([1,1],[1,2]),Placeholder(),Roll([1,1],[1,2])]
    maskList = [Placeholder(2),Placeholder(2),Mask(masks[0],masks_[0]),Mask(masks[0],masks_[0])]
    Nlayers = 4
    Hs = 10
    Ht = 10
    sList = [MLP2d(4, Hs) for _ in range(Nlayers)]
    tList = [MLP2d(4, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([4,4])
    layers = [RealNVP([2,2],
                      sList,
                      tList,
                      Gaussian([2,2]),
                      masktypelist) for _ in range(4)]
    model = HierarchyBijector(2,[[2,2] for _ in range(4)],rollList,layers,maskList,None).cuda(2)
    z = prior(2).cuda(2)
    print(z)
    x = model.inference(z,True)
    print(x)
    fLog = model._inferenceLogjac
    print(model._inferenceLogjac)
    zz = model.generate(x,True)
    print(zz)
    bLog = model._generateLogjac
    print(model._generateLogjac)

    assert_array_almost_equal(z.data.cpu().numpy(),zz.data.cpu().numpy())
    assert_array_almost_equal(fLog.cpu().numpy(),-bLog.cpu().numpy())


if __name__ == "__main__":
    #test_mera_1d()
    test_mera_2d_cuda()
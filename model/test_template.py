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

def test_tempalte_invertibleMLP():

    print("test mlp")

    gaussian = Gaussian([2])

    sList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    tList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    realNVP = RealNVP([2], sList, tList, gaussian)

    x = realNVP.prior(10)
    mask = realNVP.createMask(10)
    print("original")
    #print(x)

    z,_,_ = realNVP._generate(x,realNVP.mask,realNVP.mask_,True)

    print("Forward")
    #print(z)

    zp,_,_ = realNVP._inference(z,realNVP.mask,realNVP.mask_,True)

    print("Backward")
    #print(zp)

    assert_array_almost_equal(realNVP._generateLogjac.data.numpy(),-realNVP._inferenceLogjac.data.numpy())

    print("logProbability")
    print(realNVP.logProbability(z))

    assert_array_almost_equal(x.data.numpy(),zp.data.numpy())


def test_tempalte_invertibleCNN():

    gaussian3d = Gaussian([2,4,4])
    x3d = gaussian3d(3)
    #z3dp = z3d[:,0,:,:].view(10,-1,4,4)
    #print(z3dp)

    netStructure = [[3,2,1,1],[4,2,1,1],[3,2,1,0],[2,2,1,0]] # [channel, filter_size, stride, padding]

    sList3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]
    tList3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]

    realNVP3d = RealNVP([2,4,4], sList3d, tList3d, gaussian3d)
    mask3d = realNVP3d.createMask(3)

    print("Testing 3d")
    print("3d original:")
    #print(x3d)

    z3d,_,_ = realNVP3d._generate(x3d,realNVP3d.mask,realNVP3d.mask_,True)
    print("3d forward:")
    #print(z3d)

    zp3d,_,_ = realNVP3d._inference(z3d,realNVP3d.mask,realNVP3d.mask_,True)
    print("Backward")
    #print(zp3d)

    assert_array_almost_equal(realNVP3d._generateLogjac.data.numpy(),-realNVP3d._inferenceLogjac.data.numpy())

    print("3d logProbability")
    print(realNVP3d.logProbability(z3d))

    saveDict3d = realNVP3d.saveModel({})
    torch.save(saveDict3d, './saveNet3d.testSave')
    # realNVP.loadModel({})
    sListp3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]
    tListp3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]

    realNVPp3d = RealNVP([2,4,4], sListp3d, tListp3d, gaussian3d)
    saveDictp3d = torch.load('./saveNet3d.testSave')
    realNVPp3d.loadModel(saveDictp3d)

    zz3d = realNVPp3d.generate(x3d)
    print("3d Forward after restore")
    #print(zz3d)

    assert_array_almost_equal(x3d.data.numpy(),zp3d.data.numpy())
    assert_array_almost_equal(zz3d.data.numpy(),z3d.data.numpy())

def test_template_slice_function():
    gaussian3d = Gaussian([2,4,4])
    x = gaussian3d(3)
    #z3dp = z3d[:,0,:,:].view(10,-1,4,4)
    #print(z3dp)

    #print(x)
    netStructure = [[3,2,1,1],[4,2,1,1],[3,2,1,0],[1,2,1,0]] # [channel, filter_size, stride, padding]

    sList3d = [CNN([1,4,4],netStructure),CNN([1,4,4],netStructure),CNN([1,4,4],netStructure),CNN([1,4,4],netStructure)]
    tList3d = [CNN([1,4,4],netStructure),CNN([1,4,4],netStructure),CNN([1,4,4],netStructure),CNN([1,4,4],netStructure)]

    realNVP = RealNVP([2,4,4], sList3d, tList3d, gaussian3d)
    mask = realNVP.createMask(3)

    z = realNVP._generateWithSlice(x,0,True)
    #print(z)
    zz = realNVP._inferenceWithSlice(z,0,True)

    #print(zz)

    assert_array_almost_equal(x.data.numpy(),zz.data.numpy())
    #print(realNVP._generateLogjac.data.numpy())
    #print(realNVP._inferenceLogjac.data.numpy())
    assert_array_almost_equal(realNVP._generateLogjac.data.numpy(),-realNVP._inferenceLogjac.data.numpy())

def test_template_contraction_function():
    gaussian3d = Gaussian([2,4,4])
    x = gaussian3d(3)
    #z3dp = z3d[:,0,:,:].view(10,-1,4,4)
    #print(z3dp)

    #print(x)
    netStructure = [[3,2,1,1],[4,2,1,1],[3,2,1,0],[1,2,1,0]] # [channel, filter_size, stride, padding]

    sList3d = [CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure)]
    tList3d = [CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure),CNN([2,4,2],netStructure)]

    realNVP = RealNVP([2,4,4], sList3d, tList3d, gaussian3d)
    mask = realNVP.createMask(3,"checkerboard",1)

    z = realNVP._generateWithContraction(x,realNVP.mask,realNVP.mask_,2)
    #print(z)

    zz = realNVP._inferenceWithContraction(z,realNVP.mask,realNVP.mask_,2)
    #print(zz)

    assert_array_almost_equal(x.data.numpy(),zz.data.numpy())

if __name__ == "__main__":
    #test_tempalte_invertible()
    #test_template_slice_function()
    test_template_contraction_function()


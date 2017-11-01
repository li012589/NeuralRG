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
from model import Gaussian,MLP,RealNVP,CNN

def test_invertible():

    print("test mlp")

    gaussian = Gaussian([2])

    sList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    tList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    realNVP = RealNVP([2], sList, tList, gaussian)

    gaussian3d = Gaussian([2,4,4])
    x3d = gaussian3d(3)
    #z3dp = z3d[:,0,:,:].view(10,-1,4,4)
    #print(z3dp)

    netStructure = [[3,2,1,1],[4,2,1,1],[3,2,1,0],[2,2,1,0]] # [channel, filter_size, stride, padding]

    sList3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]
    tList3d = [CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure),CNN([2,4,4],netStructure)]

    realNVP3d = RealNVP([2,4,4], sList3d, tList3d, gaussian3d)
    mask3d = realNVP3d.createMask(3)
    #print(mask3d)

    #testCNN = CNN([1,4,4],netStructure)
    #print(testCNN.forward(x3d))


    x = realNVP.prior(10)
    mask = realNVP.createMask(10)
    print("original")
    #print(x)

    z = realNVP.generate(x)

    print("Forward")
    #print(z)
    print("logProbability")
    print(realNVP.logProbability(z))

    zp = realNVP.inference(z)

    print("Backward")
    #print(zp)

    saveDict = realNVP.saveModel({})
    torch.save(saveDict, './saveNet.testSave')
    # realNVP.loadModel({})
    sListp = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    tListp = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    realNVPp = RealNVP(2, sListp, tListp, gaussian)
    saveDictp = torch.load('./saveNet.testSave')
    realNVPp.loadModel(saveDictp)

    zz = realNVP.generate(x)
    print("Forward after restore")
    #print(zz)

    assert_array_almost_equal(x.data.numpy(),zp.data.numpy())
    assert_array_almost_equal(zz.data.numpy(),z.data.numpy())

    print("Testing 3d")
    print("3d original:")
    #print(x3d)

    z3d = realNVP3d.generate(x3d)
    print("3d forward:")
    #print(z3d)

    print("3d logProbability")
    print(realNVP3d.logProbability(z3d))

    zp3d = realNVP3d.inference(z3d)
    print("Backward")
    #print(zp3d)

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

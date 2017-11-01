if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model import RealNVPtemplate, PriorTemplate


class Gaussian(PriorTemplate):
    def __init__(self, shapeList, name="gaussian"):
        super(Gaussian, self).__init__(name)
        self.shapeList = shapeList

    def __call__(self, batchSize):
        size = [batchSize] + self.shapeList
        return Variable(torch.randn(size))

    def logProbability(self, z):
        tmp = -0.5 * (z**2)
        for i in self.shapeList:
            tmp = tmp.sum(dim=-1)
        return tmp


class MLP(nn.Module):
    def __init__(self, inNum, hideNum, name="mlp"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inNum, hideNum)
        self.fc2 = nn.Linear(hideNum, inNum)
        self.name = name

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self,inShape,netStructure,name = "cnn"):
        super(CNN, self).__init__()
        self.variableList = nn.ModuleList()
        former = inShape[0]
        self.name = name
        for layer in netStructure[:-1]:
            self.variableList.append(nn.Sequential(nn.Conv2d(former,layer[0],layer[1],layer[2],layer[3]),nn.ReLU()))
            former = layer[0]
        layer = netStructure[-1]
        self.variableList.append(nn.Sequential(nn.Conv2d(former,layer[0],layer[1],layer[2],layer[3])))
        #assert layer[0] == inshape[0]
    def forward(self,x):
        for layer in self.variableList:
            x = layer(x)
        return x

class RealNVP(RealNVPtemplate):
    def __init__(self, shapeList, sList, tList, prior):
        super(RealNVP, self).__init__(shapeList, sList, tList, prior)

    def createMask(self, batchSize):
        size = [batchSize] + self.shapeList
        size[1] = size[1] // 2
        maskOne = torch.ones(size)
        maskZero = torch.zeros(size)
        mask = torch.cat([maskOne,maskZero],1)
        self.mask = Variable(mask)
        return self.mask

    def generate(self, x):
        y, _ = self._generate(x, self.mask)
        return y

    def inference(self, x):
        y, _ = self._inference(x, self.mask)
        return y

    def logProbability(self, x):
        return self._logProbability(x, self.mask)

    def saveModel(self, saveDic):
        self._saveModel(saveDic)
        saveDic["mask"] = self.mask  # Do check if exist !!
        saveDic["shapeList"] = self.shapeList
        return saveDic

    def loadModel(self, saveDic):
        self._loadModel(saveDic)
        self.mask = saveDic["mask"]
        self.shapeList = saveDic["shapeList"]
        return saveDic


if __name__ == "__main__":

    from numpy.testing import assert_array_almost_equal,assert_array_equal
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
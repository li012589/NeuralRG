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
        variableList = nn.ModuleList()
        former = inShape[0]
        self.name = name
        for layer in netStructure:
            variableList.append(nn.Sequential(nn.Conv2d(former,layer[0],layer[1]),nn.ReLU()))
        assert layer[0] == inshape[0]
    def forward(self,x):
        for layer in variableList:
            x = layer(x)
        return x

class RealNVP(RealNVPtemplate):
    def __init__(self, shapeList, sList, tList, prior):
        super(RealNVP, self).__init__(sList, tList, prior)
        self.shapeList = shapeList

    def createMask(self, batchSize):
        size = [batchSize] + self.shapeList
        size[1] = size[1] // 2
        maskOne = torch.ones(size)
        maskZero = torch.zeros(size)
        mask = torch.cat([maskOne,maskZero],1)
        #mask = torch.cat([maskOne, maskZero], 0)
        #mask = mask.view(-1, self.shape)
        #mask = torch.cat([mask] * batchSize, 0)
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

    gaussian = Gaussian([2])

    sList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    tList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    realNVP = RealNVP([2], sList, tList, gaussian)

    '''
    gaussian3d = Gaussian([2,4,4])
    z3d = gaussian3d(10)

    netStructure = [[]]

    sList3d = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    sList3d = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    realNVP3d = RealNVP([2,3,4], sList3d, tList3d, gaussian3d)

    '''
    x = realNVP.prior(10)
    mask = realNVP.createMask(10)
    print("original")
    print(x)

    z = realNVP.generate(x)

    print("Forward")
    print(z)
    print("logProbability")
    print(realNVP.logProbability(x))

    zp = realNVP.inference(z)

    print("Backward")
    print(zp)

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
    print(zz)
    #'''

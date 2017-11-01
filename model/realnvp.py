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

    pass
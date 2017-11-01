if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model import RealNVPtemplate,PriorTemplate

class Gaussian(PriorTemplate):
    def __init__(self,numVars,name = "gaussian"):
        super(Gaussian,self).__init__(name)
        self.numVars = numVars
    def __call__(self,batchSize):
        return Variable(torch.randn(batchSize,self.numVars))
    def logProbability(self,z):
        return -0.5*(z**2)

class MLP(nn.Module):
    def __init__(self,inNum,outNum,hideNum,name="mlp"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inNum,hideNum)
        self.fc2 = nn.Linear(hideNum,outNum)
        self.name = name
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class RealNVP(RealNVPtemplate):
    def __init__(self,shape,sList,tList,prior):
        super(RealNVP,self).__init__(sList,tList,prior)
        self.shape = shape
    def createMask(self,batchSize):
        maskOne = torch.ones(self.shape//2)
        maskZero = torch.zeros(self.shape//2)
        mask = torch.cat([maskOne,maskZero],0)
        mask = mask.view(-1,self.shape)
        mask = torch.cat([mask]*batchSize,0)
        self.mask =  Variable(mask)
        return self.mask
    def generate(self,x):
        y,_ = self._generate(x,self.mask)
        return y
    def inference(self,x):
        y,_ = self._inference(x,self.mask)
        return y
    def logProbability(self,x):
        return self._logProbability(x,self.mask)
    def saveModel(self,saveDic):
        self._saveModel(saveDic)
        saveDic["mask"] = self.mask # Do check if exist !!
        saveDic["shape"] = self.shape
        return saveDic
    def loadModel(self,saveDic):
        self._loadModel(saveDic)
        self.mask = saveDic["mask"]
        self.shape = saveDic["shape"]
        return saveDic

if __name__ == "__main__":

    gaussian = Gaussian(2)

    sList = [MLP(2,2,10),MLP(2,2,10),MLP(2,2,10),MLP(2,2,10)]
    tList = [MLP(2,2,10),MLP(2,2,10),MLP(2,2,10),MLP(2,2,10)]

    realNVP = RealNVP(2,sList,tList,gaussian)

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
    #realNVP.loadModel({})
    sListp = [MLP(2,2,10),MLP(2,2,10),MLP(2,2,10),MLP(2,2,10)]
    tListp = [MLP(2,2,10),MLP(2,2,10),MLP(2,2,10),MLP(2,2,10)]

    realNVPp = RealNVP(2,sListp,tListp,gaussian)
    saveDictp = torch.load('./saveNet.testSave')
    realNVPp.loadModel(saveDictp)

    zz = realNVP.generate(x)
    print("Forward after restore")
    print(zz)
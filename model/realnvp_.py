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

class Mlp(nn.Module):
    def __init__(self,inNum,outNum,hideNum,name="mlp"):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(inNum,hideNum)
        self.fc2 = nn.Linear(hideNum,outNum)
        self.name = name
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class RealNVP(RealNVPtemplate):
    def __init__(self,sList,tList,prior):
        super(RealNVP,self).__init__(sList,tList,prior)

if __name__ == "__main__":

    gaussian = Gaussian(2)

    sList = [Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10)]
    tList = [Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10)]

    realNVP = RealNVP(sList,tList,gaussian)

    mask = torch.ByteTensor([0,1])
    x = gaussian(10)
    print("original")
    print(x)

    z,_ = realNVP.encode(x,mask)

    print("Forward")
    print(z)
    print("logProbability")
    print(realNVP.logProbability(x,mask))

    zp,_ = realNVP.decode(z,mask)

    print("Backward")
    print(zp)

    saveDict = realNVP.saveModel({})
    torch.save(saveDict, './saveNet.testSave')
    #realNVP.loadModel({})
    sListp = [Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10)]
    tListp = [Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10),Mlp(1,1,10)]

    realNVPp = RealNVP(sListp,tListp,gaussian)
    saveDictp = torch.load('./saveNet.testSave')
    realNVPp.loadModel(saveDictp)

    zz,_ = realNVP.encode(x,mask)
    print("Forward after restore")
    print(zz)
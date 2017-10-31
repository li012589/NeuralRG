import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RealNVPtemplate():
    def __init__(self,sList,tList,dataShape,prior,name=None):
        if name is None:
            self.name = "realNVP"
        else:
            self.name = name
        self.tList = tList
        self.tNumLayers = len(self.tList)
        self.sList = sList
        self.sNumLayers = len(self.sList)
        assert self.sNumLayers == self.tNumLayers
        self.NumLayers = self.sNumLayers
        self.prior=prior
    def encode(self,x,mask):
        y0 = x[mask]
        mask_ = 1-mask
        y1 = x[mask_]
        self._logjac = Variable(torch.zeros(dataShape[0]))
        for i in range(self.sNumLayers):
            if i%2 == 0:
                y1 = y1 * torch.exp(self.sList[i](y0))  + self.tList[i](y0)
                self._logjac += self.sList[i](y0).sum(dim=1)
            else:
                y0 = y0 * torch.exp(self.sList[i](y1))  + self.tList[i](y1)
                self._logjac += self.sList[i](y1).sum(dim=1)
        y = torch.cat((y0,y1),1)
        return y,mask
    def decode(self,x,mask):
        y0 = x[mask]
        mask_ = 1-mask
        y1 = x[mask_]
        for i in list(range(self.Nlayers))[::-1]:
            if (i%2==1):
                y0 = (y0 - self.tList[i](y1)) * torch.exp(-self.sList[i](y1))
            else:
                y1 = (y1 - self.tList[i](y0)) * torch.exp(-self.sList[i](y0))
        return torch.cat((y0, y1), 1)
        y = torch.cat((y0,y1),1)
        return y,mask
    def logProbability(self,x):
        z = self.forward(x)
        return self.prior.logProbability(z).sum(dim=1) + self._logjac
    def saveModel(self,saveDic):
        # save is done some where else, adding s,t to the dict
        for i in range(self.sNumLayers):
            saveDic[str(i)+'sLayer']=sList[i].state_dict()
            saveDic[str(i)+'tLayer']=tList[i].state_dict()
        return saveDic
    def loadModel(self,saveDic):
        #load is done some where else, pass the dict here.
        for i in range(self.sNumLayers):
            sList[i].load_state_dict(saveDic[str(i)+'sLayer'])
            tList[i].load_state_dict(saveDic[str(i)+'tLayer'])
        return saveDic

class PriorTemplate():
    def __init__(self,name="prior"):
        self.name = name
    def __call__(self):
        raise NotImplementedError(str(type(self)))
    def logProbability(self,x):
        raise NotImplementedError(str(type(self)))

class Gaussian(PriorTemplate):
    def __init__(self,numVars,name = "gaussian"):
        super(Gaussian,self).__init__(name)
        self.numVars = numVars
    def __call__(self,batchSize):
        return torch.randn(batchSize,self.numVars)
    def logProbability(self,z):
        return -0.5*(z**2)
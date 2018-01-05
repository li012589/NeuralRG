import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .layer import Roll, Wide2bacth, Batch2wide, Placeholder, Mask

class HierarchyBijector(nn.Module):
    def __init__(self,dimension,kernalSizeList,rollList,bijectors,maskList,prior,name = None,double = False):
        # ONLY work for one dimension!!!!
        super(HierarchyBijector,self).__init__()
        assert len(kernalSizeList) == len(bijectors)
        assert len(bijectors) == len(maskList)
        assert len(maskList) == len(rollList)

        self.bijectors = torch.nn.ModuleList(bijectors)
        self.maskList = torch.nn.ModuleList(maskList)
        self.rollList = torch.nn.ModuleList(rollList)

        self.NumLayers = len(bijectors)

        self.kernalSizeList = kernalSizeList
        self.dimension = dimension
        self.W2B = Wide2bacth(dimension)
        self.B2W = Batch2wide(dimension)

        self.prior = prior

    def inference(self,x,ifLogjac = False):
        batchSize = x.shape[0]

        if ifLogjac:
            self.register_buffer('_inferenceLogjac',torch.zeros(x.shape[0]).type(x.data.type())) # has to be pytorch 0.3 above to make this work
        for i in range(self.NumLayers):
            x,x_ = self.maskList[i].forward(x)

            shape = x.shape
            if len(shape) == 3:
                shape = shape[1:]
            else:
                shape = shape[1]
                if self.dimension == 2:
                    shape = [int((shape)**1/2) for _ in range(2)]
                    x = x.view(-1,*shape)
                elif self.dimension ==1:
                    pass
                else:
                    raise NotImplementedError("Operation corresponding to dimension is not implemneted")

            x = self.rollList[i].forward(x)

            x = self.W2B.forward(x,self.kernalSizeList[i])
            x = self.bijectors[i].inference(x,ifLogjac = ifLogjac)
            x = self.B2W.forward(x,shape)

            x = self.maskList[i].reverse(x,x_)
            #print("in "+str(i)+"th layer")
            #print(x)
            if ifLogjac:
                self._inferenceLogjac += self.bijectors[i]._inferenceLogjac.view(batchSize,-1).sum(1)

        return x

    def generate(self,x,ifLogjac = False):
        batchSize = x.shape[0]

        if ifLogjac:
            self.register_buffer('_generateLogjac',torch.zeros(x.shape[0]).type(x.data.type()))# has to be pytorch 0.3 above to make this work
        for i in reversed(range(self.NumLayers)):
            x,x_ = self.maskList[i].forward(x)

            shape = x.shape
            if len(shape) == 3:
                shape = shape[1:]
            else:
                shape = shape[1]
                if self.dimension == 2:
                    shape = [int((shape)**1/2) for _ in range(2)]
                    x = x.view(-1,*shape)
                elif self.dimension ==1:
                    pass
                else:
                    raise NotImplementedError("Operation corresponding to dimension is not implemneted")

            x = self.W2B.forward(x,self.kernalSizeList[i])
            x = self.bijectors[i].generate(x,ifLogjac = ifLogjac)
            x = self.B2W.forward(x,shape)

            x = self.rollList[i].reverse(x)
            x = self.maskList[i].reverse(x,x_)
            #print("in "+str(i)+"th layer")
            #print(x)
            if ifLogjac:
                self._generateLogjac += self.bijectors[i]._generateLogjac.view(batchSize,-1).sum(1)

        return x

    def logProbability(self,x):
        z = self.inference(x,True)
        return self.prior.probability(z) +self._inferenceLogjac

    def sample(self,batchSize):
        z = self.prior(batchSize)
        return self.generate(z)

    def saveModel(self, saveDict):
        for i, layer in enumerate(self.bijectors):
            saveDict[str(i)+"th_bijector"] = layer.saveModel({})
        return saveDict

    def loadModel(self, saveDict):
        for i,layer in enumerate(self.bijectors):
            layer.loadModel(saveDict[str(i)+"th_bijector"])
        return saveDict

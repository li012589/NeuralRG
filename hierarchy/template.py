import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#from .realnvp import RealNVP
from model import Roll, Wide2bacth, Batch2wide, Indentical


class HierarchyBijector(nn.Module):
    def __init__(self,layerSizeList,sizeList,rollList,bijectors,maskList,prior,name = None,double = False):
        # ONLY work for one dimension!!!!
        super(HierarchyBijector,self).__init__()
        assert len(layerSizeList) == len(sizeList)
        assert len(sizeList) == len(bijectors)
        assert len(bijectors) == len(maskList)
        assert maskList.shape[0] == len(rollList)

        if double:
            self.bijectors = torch.nn.ModuleList(bijectors).double()
        else:
            self.bijectors = torch.nn.ModuleList(bijectors).float()

        self.NumLayers = len(maskList)
        self.maskList = maskList

        RollLayers = []
        ReshapeLayers = []
        for i in range(self.NumLayers):
            if i % 2 == 0:
                ReshapesLayers.append(Wide2bacth(layerSizeList[i]))
            else:
                ReshapesLayers.append(Batch2wide(layerSizeList[i]))
            if i == 0:
                RollLayers.append(Identical())
            else:
                RollLayers.append(Roll(rollList[i][0],rollList[i][1]))
        self.RollLayers = torch.nn.ModuleList(RollLayers)
        self.ReshapeLayers = torch.nn.ModuleList(ReshapeLayers)
        self.register_buffer(maskList)

    def inference(self,x,ifLogjac = False):
        for layer in layers:
            pass

    def generate(self,x,ifLogjac = False):
        pass

    def cuda(self,device = None, async = False):
        pass

    def cpu(self,device = None, async = False):
        pass

    def logProbability(self,x):
        pass

    def sample(self,batchSize):
        pass

    def saveModel(self, saveDict):
        pass

    def loadModel(self, saveDict):
        pass
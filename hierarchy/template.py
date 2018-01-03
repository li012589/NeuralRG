import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#from .realnvp import RealNVP
from .layer import Roll, Wide2bacth, Batch2wide


class HierarchyBijector(nn.Module):
    def __init__(self,filterSize,modelSize,layers,maskList,prior,name = None,double = False):
        # ONLY work for one dimension!!!!
        super(TEBD,self).__init__()
        if double:
            self.layers = torch.nn.ModuleList(layers).double()
        else:
            self.layers = torch.nn.ModuleList(layers).float()
        Rolls = []
        Reshapes = []
        for i in range(len(layers)):
            if i == 0:
                Reshapes.append(Wide2bacth(filterSize))
            elif i % 2 == 0:
                Rolls.append(Roll(1,1))
            else:
                Rolls.append(Roll(1,-1))
        Reshapes.append(Batch2wide(modelSize))
        self.Rolls = torch.nn.ModuleList(Rolls)
        self.Reshapes = torch.nn.ModuleList(Reshapes)
        assert maskList.shape[0] == len(layers)
        self.register_buffer(maskList)

    def inference(self,x,ifLogjac = False):
        for layer in layers:
            x = 
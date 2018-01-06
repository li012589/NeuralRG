import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import math

from .layer import Roll, Wide2bacth, Batch2wide, Placeholder, Mask
from .template import HierarchyBijector

class MERA(HierarchyBijector):
    def __init__(self,dimension,kernalSize,configSize,bijectors,prior,name = None):
        if name is None:
            name = "MERA"
        if dimension == 1:
            depth = int(math.log(configSize,kernalSize))
            rollList = [Placeholder(),Roll(1,1)] * depth
            masks = [Variable(torch.ByteTensor([0 if i%(kernalSize**n) else 1 for i in range(configSize)])) for n in range(1,depth)]
            masks_ = [Variable(~m.data) for m in masks]
            maskList = [Placeholder(2),Placeholder(2)]
            for i in range(depth-1):
                tmp = [Mask(masks[i],masks_[i]),Mask(masks[i],masks_[i])]
                maskList += tmp
        else:
            kernalSizeS = kernalSize[0]**2
            depth = int(math.log(configSize,kernalSizeS))
            sidLen = int(math.sqrt(configSize))
            rollList = [Placeholder(),Roll([1,1],[1,2])] * depth
            masks = []
            for n in range(1,depth):
                tmp = np.zeros([kernalSize[0]**n,kernalSize[0]**n])
                tmp[0,0]=1
                tmp = np.tile(tmp,(sidLen//kernalSize[0]**n,sidLen//kernalSize[0]**n))
                masks.append(Variable(torch.from_numpy(tmp).byte()))
            masks_ = [Variable(~m.data) for m in masks]
            maskList = [Placeholder(2),Placeholder(2)]
            for i in range(depth-1):
                tmp = [Mask(masks[i],masks_[i]),Mask(masks[i],masks_[i])]
                maskList += tmp

        kernalSizeList = [kernalSize for _ in range(depth*2)]

        super(MERA,self).__init__(dimension,kernalSizeList,rollList,bijectors,maskList,prior,name)

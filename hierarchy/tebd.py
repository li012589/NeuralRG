import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .layer import Roll, Wide2bacth, Batch2wide, Placeholder, Mask
from .template import HierarchyBijector

class TEBD(HierarchyBijector):
    def __init__(self,dimension,kernalSize,depth,bijectors,prior,name = None):
        if name is None:
            name = "TEBD"
        maskList = [Placeholder(2)]*depth
        rollList = [Placeholder()]
        if dimension == 1:
            if isinstance(kernalSize,list):
                kernalSize = kernalSize[0]
            #tmp = [Roll(1,1),Roll(-1,1)] * (depth//2)
            tmp = [Roll(1,1),Placeholder()] * (depth//2)
        else:
            #tmp = [Roll([1,1],[1,2]),Roll([-1,-1],[1,2])] * (depth//2)
            tmp = [Roll([1,1],[1,2]),Placeholder()] * (depth//2)
        rollList = rollList + tmp[:-1]
        kernalSizeList = [kernalSize for _ in range(depth)]
        super(TEBD,self).__init__(dimension,kernalSizeList,rollList,bijectors,maskList,prior,name)

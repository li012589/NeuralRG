import torch
from torch import nn
import math

from .template import HierarchyBijector
from .im2col import getIndeices

class MERA(HierarchyBijector):
    def __init__(self, kernelDim, length, layerList, repeat=1, depth = None,prior=None, name = "MERA"):
        kernelSize = 2
        shape = [length,length]
        skipCheck = True
        if depth is None:
            depth = int(math.log(length,kernelSize))
            skipCheck = False
        indexList = []
        for no in range(depth):
            for _ in range(repeat):
                indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize*(kernelSize**no),kernelSize**no,0))
                indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize*(kernelSize**no),kernelSize**no,kernelSize**no))

        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        if not skipCheck:
            assert len(layerList) == len(indexIList)
            assert len(layerList) == len(indexJList)

        if kernelDim == 2:
            kernelShape = [kernelSize,kernelSize]
        elif kernelDim == 1:
            kernelShape = [kernelSize*2]

        super(MERA,self).__init__(kernelShape,indexIList,indexJList,layerList,skipCheck,prior,name)
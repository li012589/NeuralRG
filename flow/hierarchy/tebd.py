import torch
from torch import nn

from .template import HierarchyBijector
from .im2col import getIndeices

class TEBD(HierarchyBijector):
    def __init__(self,kernelDim,length,layerList,depth,prior=None,name="TEBD"):
        kernelSize = 2
        shape = [length,length]
        indexList = []
        for _ in range(depth):
            indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize,1,0))
            indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize,1,1))

        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]

        assert len(layerList) == len(indexIList)
        assert len(layerList) == len(indexJList)

        if kernelDim == 2:
            kernelShape = [kernelSize,kernelSize]
        elif kernelDim == 1:
            kernelShape = [kernelSize*2]

        super(TEBD,self).__init__(kernelShape,indexIList,indexJList,layerList,False,prior,name)
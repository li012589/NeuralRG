import torch
from torch import nn

from ..flow import Flow
from .im2col import dispatch,collect


class HierarchyBijector(Flow):
    def __init__(self, kernelShape, indexI, indexJ, layerList,prior=None,name = "HierarchyBijector"):
        super(HierarchyBijector,self).__init__(prior,name)
        assert len(layerList) == len(indexI)
        assert len(layerList) == len(indexJ)

        self.depth = len(layerList)

        self.kernelShape = kernelShape
        self.layerList = torch.nn.ModuleList(layerList)
        self.indexI = indexI
        self.indexJ = indexJ

    def inference(self,x):
        batchSize = x.shape[0]
        inferenceLogjac = x.new_zeros(x.shape[0])
        for no in range(len(self.indexI)):
            x, x_ = dispatch(self.indexI[no],self.indexJ[no],x)
            x_,logProbability = self.layerList[no].inference(x_.reshape(-1,*self.kernelShape))
            inferenceLogjac +=logProbability.view(batchSize,-1).sum(1)
            x = collect(self.indexI[no],self.indexJ[no],x,x_)
        return x,inferenceLogjac

    def generate(self,z):
        batchSize = z.shape[0]
        generateLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            z,z_ = dispatch(self.indexI[no],self.indexJ[no],z)
            z_,logProbability = self.layerList[no].generate(z_.reshape(-1,*self.kernelShape))
            generateLogjac += logProbability.view(batchSize,-1).sum(1)
            z = collect(self.indexI[no],self.indexJ[no],z,z_)
        return z,generateLogjac
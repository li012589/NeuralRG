import torch
from torch import nn

from ..flow import Flow
from .im2col import dispatch,collect


class HierarchyBijector(Flow):
    def __init__(self, kernelShape, indexI, indexJ, layerList,prior=None,name = "HierarchyBijector"):
        super(HierarchyBijector,self).__init__(prior,name)
        assert len(layerList) == len(indexI)
        assert len(layerList) == len(indexJ)

        self.kernelShape = kernelShape
        self.layerList = torch.nn.ModuleList(layerList)
        self.indexI = indexI
        self.indexJ = indexJ

    def generate(self,x,save=None):
        batchSize = x.shape[0]
        generateLogjac = x.new_zeros(x.shape[0])
        for no in range(len(self.indexI)):
            if save is not None:
                save.append(x)
            x, x_ = dispatch(self.indexI[no],self.indexJ[no],x)
            x_,logProbability = self.layerList[no].generate(x_.reshape(-1,*self.kernelShape))
            generateLogjac +=logProbability.view(batchSize,-1).sum(1)
            x = collect(self.indexI[no],self.indexJ[no],x,x_)
        return x,generateLogjac

    def inference(self,z,save=None):
        batchSize = z.shape[0]
        inferenceLogjac = z.new_zeros(z.shape[0])
        for no in reversed(range(len(self.indexI))):
            if save is not None:
                save.append(z)
            z,z_ = dispatch(self.indexI[no],self.indexJ[no],z)
            z_,logProbability = self.layerList[no].inference(z_.reshape(-1,*self.kernelShape))
            inferenceLogjac += logProbability.view(batchSize,-1).sum(1)
            z = collect(self.indexI[no],self.indexJ[no],z,z_)
        return z,inferenceLogjac
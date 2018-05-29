import torch
from torch import nn

from ..flow import Flow

class HierarchyBijector(Flow):
    def __init__(self,dims,kernelLen, maskList, layerList,prior=None,name = "HierarchyBijector"):
        super(HierarchyBijector,self).__init__(prior,name)
        self.kernalSize = [kernelLen for _ in range(dims)]
        self.layerList = layerList

    def generate(self):
        pass

    def inference(self):
        pass
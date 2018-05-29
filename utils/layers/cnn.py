import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity import Identity

class SimpleCNN2d(nn.Module):
    def __init__(self, channelList,kernelList,strideList = None, paddingList = None, dilationList=None, activation=None,name="SimpleCNN2d"):
        super(SimpleCNN2d, self).__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(kernelList)-1)]
            activation.append(nn.Tanh())
        if strideList is None:
            strideList = [1 for _ in range(len(kernelList))]
        if paddingList is None:
            paddingList = [0 for _ in range(len(kernelList))]
        if dilationList is None:
            dilationList = [1 for _ in range(len(kernelList))]

        assert len(channelList) == len(kernelList)+1
        assert len(kernelList) == len(activation)
        assert len(kernelList) == len(strideList)
        assert len(kernelList) == len(paddingList)
        assert len(kernelList) == len(dilationList)

        self.name = name

        self.layerList = nn.ModuleList()
        for no in range(len(kernelList)):
            self.layerList.append(nn.Sequential(nn.Conv2d(channelList[no], channelList[no+1], kernelList[no], strideList[no], paddingList[no]),activation[no]))

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)


    def forward(self, x):
        tmp = x
        for layer in self.layerList:
            tmp = layer(tmp)
        return tmp
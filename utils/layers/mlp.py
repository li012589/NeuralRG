import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self,dimsList,activation=None,name="SimpleMLP"):
        super(SimpleMLP,self).__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(dimsList)-2)]
            activation.append(nn.Tanh())
        self.activation = activation
        assert(len(dimsList) == len(activation)+1)
        layerList = []
        self.name = name
        for no in range(len(activation)):
            layerList.append(nn.Linear(dimsList[no],dimsList[no+1]))
            layerList.append(activation[no])
        self.layerList = torch.nn.ModuleList(layerList)

    def forward(self,x):
        tmp = x
        for layer in self.layerList:
            tmp = layer(tmp)
        return tmp

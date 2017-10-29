import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RealNVPtemplate():
    def __init__(self,layers,name=None):
        if name is None:
            self.name = "realNVP"
        else:
            self.name = name
        self.layers = layers
    def forward(self,x,mask):
        for layer in self.layers:
            x,mask = layer.forward(x,mask)
        return x,mask
    def backward(self,x,mask):
        for layer in self.layers:
            x,mask = layer.backward(x,mask)
        return x,mask

class LayerTemplate():
    def __init__(self,s,t,name="aLayer"):
        self.name = name
        self.s = s
        self.t = s
    def forward(self,x,mask):
        raise NotImplementedError(str(type(self)))
    def backward(self,x,mask):
        raise NotImplementedError(str(type(self)))
    def logProbability(self):
        pass
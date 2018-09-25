import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wide2bacthRev(nn.Module):
    def __init__(self,dims):
        super(Wide2bacthRev,self).__init__()
        if (dims == 1):
            self.pointer = "_forward2d"
        elif (dims == 2):
            self.pointer = "_forward3d"
        else:
            raise NotImplementedError("Filter size not implemneted")
    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)
    def _forward2d(self,x,kernalSize):
        x = x.reshape(-1,kernalSize)
        return x
    def _forward3d(self,x,kernalSize):
        shape = x.shape
        outSize0 = shape[1]//kernalSize[0]
        outSize1 = shape[2]//kernalSize[1]
        x = x.reshape(-1,outSize0,kernalSize[0],outSize1,kernalSize[1])
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.reshape(-1,kernalSize[0],kernalSize[1])
        return x

class Batch2wideRev(nn.Module):
    def __init__(self,dims):
        super(Batch2wideRev,self).__init__()
        if (dims == 1):
            self.pointer = "_forward2d"
        elif (dims == 2):
            self.pointer = "_forward3d"
        else:
            raise NotImplementedError("Filter size not implemneted")
    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)
    def _forward2d(self,x,kernalSize):
        x = x.reshape(-1,kernalSize)
        return x
    def _forward3d(self,x,kernalSize):
        shape = x.shape
        outSize0 = kernalSize[0]//shape[1]
        outSize1 = kernalSize[1]//shape[2]
        x = x.reshape(-1,outSize0,outSize1,shape[1],shape[2])
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.reshape(-1,kernalSize[0],kernalSize[1])
        return x

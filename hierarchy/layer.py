import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model import MLP

class Placeholder(nn.Module):
    def __init__(self,output = 1):
        super(Placeholder,self).__init__()
        if output == 1:
            self.pointer = "forward1"
            self.rpointer = "reverse1"
        else:
            self.pointer = "forward2"
            self.rpointer = "reverse2"
    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)
    def reverse(self,*args,**kwargs):
        return getattr(self,self.rpointer)(*args,**kwargs)
    def forward1(self,x):
        return x
    def reverse1(self,x):
        return x
    def forward2(self,x):
        return x,None
    def reverse2(self,x,other):
        return x

class debugRealNVP(nn.Module):
        def __init__(self):
            super(debugRealNVP,self).__init__()
        def forward(self,x,ifLogjac):
            return x
        def generate(self,x,ifLogjac):
            return x
        def inference(self,x,ifLogjac):
            return x

class Roll(nn.Module):
    def __init__(self,step,axis):
        super(Roll,self).__init__()
        if not isinstance(step,list):
            assert not isinstance(axis,list)
            step = [step]
            axis = [axis]
        assert len(step) == len(axis)
        self.step = step
        self.axis = axis
    def forward(self,x):
        shape = x.shape
        for i,s in enumerate(self.step):
            if s >=0:
                x1 = x.narrow(self.axis[i],0,s)
                x2 = x.narrow(self.axis[i],s,shape[self.axis[i]]-s)
            else:
                x2 = x.narrow(self.axis[i],shape[self.axis[i]]+s,-s)
                x1 = x.narrow(self.axis[i],0,shape[self.axis[i]]+s)
            x = torch.cat([x2,x1],self.axis[i])
        return x
    def reverse(self,x):
        shape = x.shape
        for i,s in enumerate(self.step):
            s = -s
            if s >=0:
                x1 = x.narrow(self.axis[i],0,s)
                x2 = x.narrow(self.axis[i],s,shape[self.axis[i]]-s)
            else:
                x2 = x.narrow(self.axis[i],shape[self.axis[i]]+s,-s)
                x1 = x.narrow(self.axis[i],0,shape[self.axis[i]]+s)
            x = torch.cat([x2,x1],self.axis[i])
        return x
class Mask(nn.Module):
    def __init__(self,mask,mask_):
        super(Mask,self).__init__()
        self.register_buffer("mask",mask)
        self.register_buffer("mask_",mask_)
    def forward(self,x,size=None):
        batchSize = x.shape[0]
        if size is None:
            size = [batchSize,-1]
        else:
            size = [-1] + size
        return torch.masked_select(x,self.mask).view(*size),torch.masked_select(x,self.mask_).view(batchSize,-1)
    def reverse(self,x,x_):
        batchSize = x.shape[0]
        if x.is_cuda:
            output = Variable(torch.zeros(batchSize,*self.mask.shape).cuda(x.get_device()).type(x.data.type()))
        else:
            output = Variable(torch.zeros(batchSize,*self.mask.shape).type(x.data.type()))
        output.masked_scatter_(self.mask,x)
        output.masked_scatter_(self.mask_,x_)
        return output

class Wide2bacth(nn.Module):
    def __init__(self,dims):
        super(Wide2bacth,self).__init__()
        if (dims == 1):
            self.pointer = "_forward2d"
        elif (dims == 2):
            self.pointer = "_forward3d"
        else:
            raise NotImplementedError("Filter size not implemneted")
    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)
    def _forward2d(self,x,kernalSize):
        x = x.view(-1,kernalSize)
        return x
    def _forward3d(self,x,kernalSize):
        shape = x.shape
        outSize0 = shape[1]//kernalSize[0]
        outSize1 = shape[2]//kernalSize[1]
        x = x.view(-1,outSize0,kernalSize[0],outSize1,kernalSize[1])
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.view(-1,kernalSize[0],kernalSize[1])
        return x

class Batch2wide(nn.Module):
    def __init__(self,dims):
        super(Batch2wide,self).__init__()
        if (dims == 1):
            self.pointer = "_forward2d"
        elif (dims == 2):
            self.pointer = "_forward3d"
        else:
            raise NotImplementedError("Filter size not implemneted")
    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)
    def _forward2d(self,x,kernalSize):
        x = x.view(-1,kernalSize)
        return x
    def _forward3d(self,x,kernalSize):
        shape = x.shape
        outSize0 = kernalSize[0]//shape[1]
        outSize1 = kernalSize[1]//shape[2]
        x = x.view(-1,outSize0,shape[1],outSize1,shape[2])
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.view(-1,kernalSize[0],kernalSize[1])
        return x

class MLP2d(MLP):
    def __init__(self,*args,**kwargs):
        super(MLP2d,self).__init__(*args,**kwargs)
    def forward(self,x):
        shape = x.shape
        x = x.view(shape[0],-1)
        x = super(MLP2d,self).forward(x)
        return x.view(shape)
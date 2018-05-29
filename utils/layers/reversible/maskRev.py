import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskRev(nn.Module):
    def __init__(self,mask,mask_):
        super(MaskRev,self).__init__()
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
        output = x.new_zeros([batchSize,*self.mask.shape])
        output.masked_scatter_(self.mask,x)
        output.masked_scatter_(self.mask_,x_)
        return output

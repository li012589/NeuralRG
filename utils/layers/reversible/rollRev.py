import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...roll import roll

class RollRev(nn.Module):
    def __init__(self,step,axis):
        super(RollRev,self).__init__()
        if not isinstance(step,list):
            assert not isinstance(axis,list)
            step = [step]
            axis = [axis]
        assert len(step) == len(axis)
        self.step = step
        self.axis = axis
    def forward(self,x):
        return roll(x,self.step,self.axis)

    def reverse(self,x):
        return roll(x,[-i for i in self.step],self.axis)

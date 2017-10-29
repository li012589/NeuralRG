import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RealNVPtemplate():
    def __init__(self,s,t,name=None):
        if name is None:
            self.name ='Nvars'+str(self.Nvars) \
                   +'Nlayers'+str(self.Nlayers) \
                   +'Hs'+str(self.Hs)  \
                   +'Ht'+str(self.Ht) \
                   +'.realnvp'
        else:
            self.name = name
        self.s = s
        self.t = t
    def forward(z,mask)
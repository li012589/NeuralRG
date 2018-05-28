import torch
from torch import nn

from model import Flow

class RNVP(Flow):
    def __init__(self, maskList, tList, sList, prior = None, name = "RNVP"):
        super(RNVP,self).__init__(prior,name)
        self.maskList = nn.Parameter(maskList)
        self.tList = tList
        self.sList = sList
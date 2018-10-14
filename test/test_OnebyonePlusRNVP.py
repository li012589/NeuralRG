from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source

def test_bijective():
    p = source.Gaussian([3,2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f = flow.OnebyonePlusRNVP(maskList, [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),utils.ScalableTanh(12)]) for _ in range(4)],2,2,3,p)
    bijective(f,decimal=3)

def test_saveload():
    p = source.Gaussian([3,2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    f = flow.OnebyonePlusRNVP(maskList, [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),utils.ScalableTanh(12)]) for _ in range(4)],2,2,3,p)

    p = source.Gaussian([3,2,2])

    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.reshape(1,1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    blankf = flow.OnebyonePlusRNVP(maskList, [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([12,32,32,12],[nn.ELU(),nn.ELU(),utils.ScalableTanh(12)]) for _ in range(4)],2,2,3,p)

    saveload(f,blankf,decimal=3)

if __name__ == "__main__":
    test_bijective()

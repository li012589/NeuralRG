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
    p = source.Gaussian([1,4,4])

    BigList = []
    for _ in range(2*2*2):
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
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,prior = p)
    bijective(t)

def test_bijective_different_depth():
    p = source.Gaussian([1,4,4])

    BigList = []
    for _ in range(2*2*2):
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
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,depth = 1,prior = p)
    bijective(t)

def test_saveload():
    p = source.Gaussian([1,4,4])

    BigList = []
    for _ in range(2*2*2):
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
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,prior = p)

    p = source.Gaussian([1,4,4])

    BigList = []
    for _ in range(2*2*2):
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
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    blankt = flow.MERA(2,length,layers,repeat,prior = p)
    saveload(t,blankt)
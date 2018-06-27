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
import train

def test_bijective():
    p = source.Gaussian([4,4])

    BigList = []
    for _ in range(2*2*2):
        maskList = []
        for n in range(4):
            if n %2==0:
                b = torch.zeros(1,4)
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
                b.zero_()[:,i] = 1
                b=b.view(1,2,2)
            else:
                b = 1-b
            maskList.append(b)
        maskList = torch.cat(maskList,0).to(torch.float32)
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,p)
    def op(x):
        return -x

    sym = [op]
    m = train.Symmetrized(t, sym)
    z = m.prior.sample(100)
    xz1,_ = m.inverse(z)
    xz2,_ = m.inverse(z)
    p1 = m.logProbability(xz1)
    p2 = m.logProbability(xz2)

    z1,_ = m.forward(xz1)
    xz1p,_ = m.inverse(z1)

    assert ((xz1 == xz2).sum() + (xz1 == -xz2).sum()) == 100*4*4
    assert_array_almost_equal(p1.detach().numpy(),p2.detach().numpy(),decimal=5)
    assert_array_almost_equal(np.fabs(xz1.detach().numpy()),np.fabs(xz1p.detach().numpy()))


def test_saveload():
    p = source.Gaussian([4,4])

    BigList = []
    for _ in range(2*2*2):
        maskList = []
        for n in range(4):
            if n %2==0:
                b = torch.zeros(1,4)
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
                b.zero_()[:,i] = 1
                b=b.view(1,2,2)
            else:
                b = 1-b
            maskList.append(b)
        maskList = torch.cat(maskList,0).to(torch.float32)
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,p)
    def op(x):
        return -x

    sym = [op]
    m = train.Symmetrized(t, sym)

    p = source.Gaussian([4,4])

    BigList = []
    for _ in range(2*2*2):
        maskList = []
        for n in range(4):
            if n %2==0:
                b = torch.zeros(1,4)
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
                b.zero_()[:,i] = 1
                b=b.view(1,2,2)
            else:
                b = 1-b
            maskList.append(b)
        maskList = torch.cat(maskList,0).to(torch.float32)
        BigList.append(maskList)

    layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

    length = 4
    repeat = 2

    t = flow.MERA(2,length,layers,repeat,p)
    def op(x):
        return -x

    sym = [op]
    blankm = train.Symmetrized(t, sym)

    z = m.prior.sample(100)
    xz1,_ = m.inverse(z)

    d = m.save()
    torch.save(d,"testsaving.saving")
    dd = torch.load("testsaving.saving")
    blankm.load(dd)
    xz2,_ = blankm.inverse(z)
    p1 = m.logProbability(xz1)
    p2 = blankm.logProbability(xz2)
    assert ((xz1 == xz2).sum() + (xz1 == -xz2).sum()) == 100*4*4
    assert_array_almost_equal(p1.detach().numpy(),p2.detach().numpy(),decimal=5)

if __name__ == "__main__":
    test_bijective()
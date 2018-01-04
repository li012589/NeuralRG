import os
import sys
sys.path.append(os.getcwd())

from profilehooks import profile

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal
from model import Gaussian,MLP,RealNVP,CNN,Squeezing
from hierarchy import Roll,Wide2bacth,Batch2wide,MLP2d

from subprocess import Popen, PIPE
import pytest

noCuda = 0
try:
    p  = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode", "--format=csv,noheader,nounits"], stdout=PIPE)
except OSError:
    noCuda = 1

maxGPU = 0
if noCuda == 0:
    try:
        p = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader,nounits')
        i = p.read().split('\n')
        maxGPU = int(i[-2])+1
    except OSError:
        noCuda = 1

skipIfNoCuda = pytest.mark.skipif(noCuda == 1,reason = "No cuda insatllation, found through nvidia-smi")
skipIfOnlyOneGPU = pytest.mark.skipif(maxGPU < 2,reason = "Only one gpu")

def test_squeeze():
    t = torch.arange(0,16,1).view(1,1,4,4)
    print(t)
    s = Squeezing(1/2)
    s2 = Squeezing(2)
#    import pdb
 #   pdb.set_trace()
    t2 = s(t)
    print(t2)
    t3 = s2(t2)
    print(t3)
    t = t.numpy()
    t3 = t3.numpy()
    assert_array_almost_equal(t,t3)

def test_Roll():
    #t = np.random.randn(2,4,4)
    t = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    r1 = np.roll(t,1,0)
    r2 = np.roll(t,1,1)
    r3 = np.roll(t,-1,0)
    r4 = np.roll(t,-1,1)

    tt = torch.from_numpy(t)
    tt = tt.view(-1,4,4)
    ll1 = Roll(1,1)
    l1 = ll1.forward(tt)
    l1p = ll1.reverse(l1)
    ll2 = Roll(1,2)
    l2 = ll2.forward(tt)
    l2p = ll2.reverse(l2)
    ll3 = Roll(-1,1)
    l3 = ll3.forward(tt)
    l3p = ll3.reverse(l3)
    ll4 = Roll(-1,2)
    l4 = ll4.forward(tt)
    l4p = ll4.reverse(l4)

    #print(tt)
    #print(l4)
    #print(l4p)

    assert_array_almost_equal(r1,l3.numpy()[0])
    assert_array_almost_equal(r2,l4.numpy()[0])
    assert_array_almost_equal(r3,l1.numpy()[0])
    assert_array_almost_equal(r4,l2.numpy()[0])
    assert_array_equal(l1p.numpy(),l2p.numpy())
    assert_array_equal(l3p.numpy(),l2p.numpy())
    assert_array_equal(l4p.numpy(),l2p.numpy())
def test_Wide2bacth():
    a = torch.FloatTensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8]])
    print(a)
    l = Wide2bacth(1)
    b = l.forward(a,2)
    print(b)
    assert b.shape[0] == 8
    assert b.shape[1] == 2
    lb = Batch2wide(1)
    ab = lb.forward(b,8)
    print(ab)
    assert_array_equal(a.numpy(),ab.numpy())

    aa = torch.FloatTensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[2,3,4,5],[6,7,8,9],[10,11,12,13],[14,15,16,17]]])
    print(aa)
    ll = Wide2bacth(2)
    bb = ll.forward(aa,[2,2])
    print(bb)
    assert bb.shape[0] == 8
    assert bb.shape[1] == 2
    assert bb.shape[2] == 2
    llb = Batch2wide(2)
    aab = llb.forward(bb,[4,4])
    print(aab)
    assert_array_equal(aa.numpy(),aab.numpy())

def test_MLP2d():
    aa = Variable(torch.randn(2,2,2))
    model = MLP2d(4,10)
    bb = model.forward(aa)
    print(bb)
    assert bb.shape[0] == 2
    assert bb.shape[1] == 2
    assert bb.shape[2] == 2

if __name__ == "__main__":
    #test_Wide2bacth()
    #test_Roll()
    test_MLP2d()
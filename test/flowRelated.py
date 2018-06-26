import torch
import numpy as np
from numpy.testing import assert_array_almost_equal,assert_array_equal

def bijective(flow,batch=100):
    x,p = flow.sample(batch)
    z,ip = flow.inference(x)
    xz,gp = flow.generate(z)
    op = flow.prior.logProbability(z)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy())
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy())
    assert_array_almost_equal(p.detach().numpy(),(op+gp).detach().numpy())

def saveload(flow,blankFlow,batch=100):
    x,p = flow.sample(batch)
    z,ip = flow.inference(x)
    d = flow.save()
    torch.save(d,"testsaving.saving")
    dd = torch.load("testsaving.saving")
    blankFlow.load(dd)
    op = blankFlow.prior.logProbability(z)
    xz,gp = blankFlow.generate(z)
    assert_array_almost_equal(x.detach().numpy(),xz.detach().numpy())
    assert_array_almost_equal(ip.detach().numpy(),-gp.detach().numpy())
    assert_array_almost_equal(p.detach().numpy(),(op+gp).detach().numpy())
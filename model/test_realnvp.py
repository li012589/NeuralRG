import torch 
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np 

from model.realnvp import RealNVP 

def test_invertible():

    Nsamples = 1000
    Nvars = 4

    model = RealNVP(Nvars)
    
    x = Variable(torch.randn(Nsamples, Nvars))
    z = model.forward(x)

    assert(np.allclose(x.data.numpy(), model.backward(z).data.numpy(), atol=1E-6))

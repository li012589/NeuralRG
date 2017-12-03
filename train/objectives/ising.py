import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh 

from .template import Target
from .lattice import Hypercube

class Ising(Target):

    def __init__(self, L, d, K):
        super(Ising, self).__init__(L**d,'Ising')

        lattice = Hypercube(L, d)
        A = lattice.Adj * K
    
        w, v = eigh(A)    
        d = 1.0-w.min()
        self.Lambda = Variable(torch.from_numpy(w+d).view(-1,len(w)),  requires_grad=False)
        self.VT = Variable( torch.from_numpy(v.transpose()), requires_grad=False)
    
        #print (self.d)
        #print (v)
        #print (self.Lambda)
        print (self.VT)

    def energy(self, x): # actually logp
        return -0.5*(x**2/self.Lambda).sum(dim=1) \
        + torch.log(torch.cosh(torch.mm(x, self.VT))).sum(dim=1)
    
    def measure(self, x):
        p = torch.sigmoid(2.*torch.mm(Variable(x), self.VT)) 
        #sample spin
        #s = 2*torch.bernoulli(p).data.numpy()-1
        #return (s.mean(axis=1))**2
        #for i in range(s.shape[0]):
        #    print (' '.join(map(str, s[i,:])))
 
        #improved estimato 
        s = 2.*p.data.numpy() - 1. 
        return (s.mean(axis=1))**2 - (s**2).sum(axis=1)/self.nvars**2  +1./self.nvars 

if __name__=='__main__':
    torch.manual_seed(42)
    K = -1.0 
    ising = Ising(K) 
    x = Variable(torch.randn(10, 2).double())
    print (x)
    print (ising.energy(x))
    print (ising.measure(x.data))


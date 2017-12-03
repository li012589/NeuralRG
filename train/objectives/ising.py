import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh 
from .template import Target

class Ising(Target):

    def __init__(self):
        super(Ising, self).__init__(2,'Ising')
        A = np.array([[0., 1.],
                      [1., 0.]], 
                      dtype=np.float64)
    
        w, v = eigh(A)    
        self.d = 1-w.min()
        self.Lambda = Variable(torch.from_numpy(w+self.d).view(-1,len(w)),  requires_grad=False)
        self.VT = Variable( torch.from_numpy(v.transpose()), requires_grad=False)
    
        #print (self.d)
        #print (v)
        #print (self.Lambda)
        #print (self.VT)

    def energy(self, x): # actually logp
        return -0.5*(x**2/self.Lambda).sum(dim=1) \
        + torch.log(torch.cosh(torch.mm(x, self.VT))).sum(dim=1)
    
    def measure(self, x):
        p = torch.sigmoid(2.*torch.mm(Variable(x), self.VT)) 
        s = 2*torch.bernoulli(p).data.numpy()-1

        return (s.sum(axis=1))**2

if __name__=='__main__':
    torch.manual_seed(42)
    ising = Ising() 
    x = Variable(torch.randn(10, 2).double())
    print (x)
    print (ising.energy(x))
    print (ising.measure(x.data))


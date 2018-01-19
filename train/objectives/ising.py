import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh, inv, det 

from .template import Target
from .lattice import Hypercube

class Ising(Target):

    def __init__(self, L, d, T, cuda=None, double = False):
        super(Ising, self).__init__(L**d,'Ising')

        self.lattice = Hypercube(L, d, 'periodic')
        self.Nvars = self.lattice.Nsite
        self.K = self.lattice.Adj/T
    
        w, v = eigh(self.K)    
        offset = 0.1-w.min()
        self.K += np.eye(w.size)*offset
        print (0.5*self.Nvars *(np.log(4.)-offset - np.log(2.*np.pi)) - 0.5*np.log(det(self.K)))
        if not double:
            self.Kinv = Variable(torch.from_numpy(inv(self.K)).float(), requires_grad=False)
        else:
            self.Kinv = Variable(torch.from_numpy(inv(self.K)), requires_grad=False)
        #self.VT = Variable( torch.from_numpy(v.transpose()), requires_grad=False)
        if cuda is not None:
            #self.VT = self.VT.cuda(cuda)
            self.Kinv = self.Kinv.cuda(cuda)
        #print (self.d)
        #print (v)
        #print (self.Lambda)
        #print (self.VT)
        #print (self.Kinv)

    def energy(self, x): # actually logp

        
        #return -0.5*(x**2).sum(dim=1) \
        #+ torch.log(torch.cosh(self.beta*torch.mm(x, self.VT))).sum(dim=1)
        return -0.5*(torch.mm(x.view(-1, self.Nvars),self.Kinv) * x.view(-1, self.Nvars)).sum(dim=1) \
        + torch.log(torch.cosh(self.beta*x.view(-1, self.Nvars))).sum(dim=1)
    
    def measure(self, x):
        p = torch.sigmoid(2.*x).view(-1, self.Nvars)
        #sample spin
        #s = 2*torch.bernoulli(p).data.numpy()-1
        #sf = (s.mean(axis=1))**2
        #for i in range(s.shape[0]):
        #    print (' '.join(map(str, s[i,:])))
 
        #improved estimator
        s = 2.*p.data.cpu().numpy() - 1. 
        #en = -(np.dot(s, self.K) * s).mean(axis= 1) # energy
        sf = (s.mean(axis=1))**2 - (s**2).sum(axis=1)/self.nvars**2  +1./self.nvars #structure factor
        return  sf
    def set_beta(self, beta):
        self.beta = beta 

if __name__=='__main__':
    torch.manual_seed(42)
    L = 8
    d = 2
    T = 2.269185314213022
    ising = Ising(L, d, T) 
    #x = Variable(torch.randn(10, 2).double())
    #print (x)
    #print (ising.energy(x))
    #print (ising.measure(x.data))


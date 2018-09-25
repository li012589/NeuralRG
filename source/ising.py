import numpy as np
import torch
import torch.nn.functional as F

import scipy.sparse as sps
from scipy.linalg import eigh, inv, det 
from numpy import zeros
import math

from .source import Source
from utils import roll

class Lattice:
    def __init__(self,L, d, BC='periodic'):
        self.L = L 
        self.d = d
        self.shape = [L]*d 
        self.Nsite = L**d 
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L; 
        if (coord[d]<0): coord[d] += self.L; 

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = zeros(self.d, int) 
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord 

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L; 
            idx += coord[d]
        return idx 

class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0

class Ising(Source):
    def __init__(self,L,d,T,name = None):
        if name is None:
            name = "Ising_l"+str(L)+"_d" +str(d)+"_t"+str(T)
        super(Ising,self).__init__([L**d],name)
        self.beta = 1.0
        self.lattice = Hypercube(L, d, 'periodic')
        self.K = self.lattice.Adj/T
    
        w, v = eigh(self.K)    
        offset = 0.1-w.min()
        self.K += np.eye(w.size)*offset
        sign, logdet = np.linalg.slogdet(self.K)
        #print (sign)
        #print (0.5*self.nvars[0] *(np.log(4.)-offset - np.log(2.*np.pi)) - 0.5*logdet)
        Kinv = torch.from_numpy(inv(self.K)).to(torch.float32)
        self.register_buffer("Kinv",Kinv)

    def energy(self,x):
        return -(-0.5*(torch.mm(x.reshape(-1, self.nvars[0]),self.Kinv) * x.reshape(-1, self.nvars[0])).sum(dim=1) \
        + (torch.nn.Softplus()(2.*self.beta*x.reshape(-1, self.nvars[0])) - self.beta*x.reshape(-1, self.nvars[0]) - math.log(2.)).sum(dim=1))

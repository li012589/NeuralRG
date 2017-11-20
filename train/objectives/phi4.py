import torch
from torch.autograd import Variable
import numpy as np
from .template import Target

class Phi4(Target):
    def __init__(self,l,dims,kappa,lamb,hoppingTable=None,name=None):
        if name is None:
            name = "phi4_l"+str(l)+"_d"+str(dims)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            pass
        n = l**dims
        super(Phi4,self).__init__(n,name)
        self.dims = dims
        self.nvars = n
        self.l = l
        self.kappa = kappa
        self.lamb = lamb
        if hoppingTable is None:
            self.hoppingTable = self.createTable()
        else:
            self.hoppingTable = hoppingTable
        super(Phi4, self).__init__(self.nvars,self.name)
    def energy(self,z):
        S = Variable(torch.zeros(z[:,0].shape).double())
        for i in range(self.nvars):
            tmp = Variable(torch.zeros(z[:,0].shape).double())
            for j in range(self.dims):
                #print(z[:,self.hoppingTable[i][j*2]])
                tmp += z[:,self.hoppingTable[i][j*2]]
            #print("tmp: ",tmp)
            S += -2*self.kappa*tmp*z[:,i]
        S+=torch.sum(z**2,1)+self.lamb*torch.sum((z**2-1)**2,1)
        return S
    def createTable(self):
        hoppingTable = []
        for i in range(self.nvars):
            LK = self.nvars
            y = i
            hoppingTable.append([])
            for j in reversed(range(self.dims)):
                LK = int(LK/self.l)
                xk = int(y/LK)
                y = y-xk*LK
                if xk < self.l-1:
                    hoppingTable[i].append(i + LK)
                else:
                    hoppingTable[i].append(i + LK*(1-self.l))
                if xk > 0:
                    hoppingTable[i].append(i - LK)
                else:
                    hoppingTable[i].append(i-LK*(1-self.l))
        return hoppingTable
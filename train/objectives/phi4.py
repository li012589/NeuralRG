import torch
from torch.autograd import Variable
import numpy as np
from .template import Target

class Phi4(Target):
    def __init__(self,l,dims,kappa,lamb,hoppingTable=None,name=None):
        self.dims = dims
        self.l = l
        self.nvars = l**dims
        self.kappa = kappa
        self.lamb = lamb
        if name is None:
            self.name = "phi_"+str(self.nvars)+"n_"+str(self.l)+"l_"+str(self.dims)+"dim_"+str(self.kappa)+"kappa_"+str(self.lamb)+"lambda"
        else:
            self.name = name
        if hoppingTable is None:
            self.hoppingTable = self.createTable()
        else:
            self.hoppingTable = hoppingTable
        super(Phi4, self).__init__(self.nvars,self.name)
    def energy(self,z):
        if isinstance(z.data,torch.DoubleTensor):
            S = Variable(torch.zeros(z[:,0].data.shape).double())
            tmp = Variable(torch.zeros(z[:,0].data..shape).double())
        else:
            S = Variable(torch.zeros(z[:,0].data.shape))
            tmp = Variable(torch.zeros(z[:,0].data.shape))
        for i in range(self.nvars):
            tmp.data.zero_()
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
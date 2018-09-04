import numpy as np
import torch

from .source import Source
from utils import roll

def no2ij(n,dList):
    cood =  []
    d = len(dList)
    for dim in reversed(range(len(dList))):
        L = dList[dim]
        tmp1 = (int(n/(L**dim)))
        n -= tmp1*(L**dim)
        cood.append(tmp1)
    return cood

def ij2no(cood,dList):
    n = 0
    d = len(dList)
    for dim in reversed(range(len(dList))):
        L = dList[dim]
        n += (cood[d-dim-1])*(L**dim)
    return n

def Kijbuilder(dList,k,lamb,peridic=True,skip=None):
    maxNo = 1
    for d in dList:
        maxNo *= d
    Kij = torch.zeros([maxNo]*2)
    for no in range(maxNo):
        cood = no2ij(no,dList)
        for i in range(len(cood)):
            if i in skip:
                continue
            coodp = cood.copy()
            coodp[i] = (cood[i]+1)%dList[i]
            Kij[no,ij2no(coodp,dList)] = k
            coodp[i] = (cood[i]-1)%dList[i]
            Kij[no,ij2no(coodp,dList)] = k
    tmp = torch.diag(torch.tensor([lamb]*(maxNo),dtype=torch.float32))
    return Kij+tmp


class Phi4c(Source):
    def __init__(self,l,dims,kappa,lamb,name = None):
        if name is None:
            self.name = "phi4_l"+str(l)+"_d"+str(dims)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            self.name = name

        nvars = [2]
        for _ in range(dims):
            nvars += [l]
        super(Phi4c,self).__init__(nvars,name)

        self.kappa = torch.nn.Parameter(torch.tensor([kappa],dtype=torch.float32),requires_grad = False)
        self.lamb = torch.nn.Parameter(torch.tensor([lamb],dtype=torch.float32),requires_grad = False)
        self.dims = dims

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)

    def energy(self,x):
        S = 0
        for i in range(self.dims):
            S += x*roll(x,[1],[i+2])
            #S += x*roll(x,[-1],[i+1])
        term1 = x[:,0]**2+x[:,1]**2
        term2 = (term1-1)**2
        for _ in range(self.dims):
            S = S.sum(-1)
            term1 = term1.sum(-1)
            term2 = term2.sum(-1)
        S *= -2*self.kappa
        term2 *= self.lamb
        out = S[:,0]-S[:,1]+term2+term1
        return out


class Phi4complex(Source):
    def __init__(self,l,dims,kappa,lamb,name = None):
        if name is None:
            self.name = "phi4_l"+str(l)+"_d"+str(dims)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            self.name = name

        nvars = [2]
        for _ in range(dims):
            nvars += [l]
        super(Phi4complex,self).__init__(nvars,name)

        self.K = Kijbuilder([2]+[l]*dims,kappa,0,skip=[0])
        maxNo = self.K.shape[0]
        diag = torch.diagonal(self.K)
        self.K[int(maxNo/2):,int(maxNo/2):] = -self.K[int(maxNo/2):,int(maxNo/2)]
        self.K += torch.diag(torch.tensor([1]*(maxNo),dtype=torch.float32))

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)

    def energy(self,x):
        out = torch.mm(torch.mm(x.reshape(1,-1),self.K),x.reshape(-1,1))
        return out

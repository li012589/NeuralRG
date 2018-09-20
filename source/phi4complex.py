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

def Kijbuilder(dList,k,lamb,skip=[]):
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
            Kij[no,ij2no(coodp,dList)] += k
            coodp[i] = (cood[i]-1)%dList[i]
            Kij[no,ij2no(coodp,dList)] += k
    tmp = torch.diag(torch.tensor([lamb]*(maxNo),dtype=torch.float32))
    return Kij+tmp

'''
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

        self.lamb = lamb
        K = Kijbuilder([2]+[l]*dims,-kappa,0,skip=[0])
        maxNo = K.shape[0]
        K[int(maxNo/2):,int(maxNo/2):] = -K[int(maxNo/2):,int(maxNo/2):]
        Kp = torch.diag(torch.tensor([1]*maxNo,dtype=torch.float32))
        K += Kp
        self.register_buffer("K",K)

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)

    def energy(self,x):
        batchSize = x.shape[0]
        out = torch.matmul(torch.matmul(x.reshape(batchSize,1,-1),self.K),x.reshape(batchSize,-1,1)).reshape(batchSize)
        out += (((x.reshape(batchSize,-1)*x.reshape(batchSize,-1)).reshape(batchSize,2,-1).sum(-2)-1)**2).sum(-1)*self.lamb
        return out
'''

class Phi4(Source):
    def __init__(self,n,l,dims,kappa,lamb,name = None):
        if name is None:
            self.name = "phi4_l"+str(l)+"_d"+str(dims)+"_kappa"+str(kappa)+"_lamb"+str(lamb)
        else:
            self.name = name

        nvars = [n]
        for _ in range(dims):
            nvars += [l]
        super(Phi4,self).__init__(nvars,name)

        Kt = Kijbuilder([l]*dims,-kappa,1)
        K = Kt
        for _ in range(n-1):
            tmp1 = torch.zeros([K.shape[0],Kt.shape[1]])
            tmp1 = torch.cat([K,tmp1],1)
            tmp2 = torch.zeros([Kt.shape[0],K.shape[1]])
            tmp2 = torch.cat([tmp2,Kt],1)
            K = torch.cat([tmp1,tmp2],0)
        self.register_buffer("K",K)
        self.lamb = lamb
        self.kappa = kappa
        self.channel = n

    def sample(self, batchSize, thermalSteps = 50, interSteps=5, epsilon=0.1):
        return self._sampleWithHMC(batchSize,thermalSteps,interSteps, epsilon)

    def energy(self,x):
        batchSize = x.shape[0]
        out = torch.matmul(torch.matmul(x.reshape(batchSize,1,-1),self.K),x.reshape(batchSize,-1,1)).reshape(batchSize)
        out += (((x**2).sum(1)-1)**2).reshape(batchSize,-1).sum(-1)*self.lamb
        return out

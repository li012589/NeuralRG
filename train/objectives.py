import torch 
import numpy as np 

class Ring2D(object):

    def __init__(self):
        self.name = 'Ring2d'

    def __call__(self, x):
        return -(torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32

class Ring5(object):

    def __init__(self):
        self.name = 'Ring5'

    def __call__(self, x):
        x2 = torch.sqrt((x**2).sum(dim=1))
        u1 = (x2 - 1.) **2 /0.04
        u2 = (x2 - 2.) **2 /0.04
        u3 = (x2 - 3.) **2 /0.04
        u4 = (x2 - 4.) **2 /0.04
        u5 = (x2 - 5.) **2 /0.04

        u1 = u1.view(-1, 1)
        u2 = u2.view(-1, 1)
        u3 = u3.view(-1, 1)
        u4 = u4.view(-1, 1)
        u5 = u5.view(-1, 1)

        u = torch.cat((u1, u2, u3, u4, u5), dim=1)
        return -torch.min(u, dim=1)[0]


class Wave(object):

    def __init__(self):
        self.name = "Wave"

    def __call__(self, x):
        w = torch.sin(np.pi*x[:, 0]/2.)
        return -0.5*((x[:, 1] -w)/0.4)**2

class phi4(object):
    def __init__(self,l,dims,kappa,lamb,hoppingTable,name=None):
        self.dims = dims
        self.n = l**dims
        self.hoppingTable = hoppingTable
        self.kappa = kappa
        self.lamb = lamb
        if name is None:
            self.name = "phi_"+str(self.n)+"n_"+str(self.l)+"l_"+str(self.dims)+"dim_"+str(self.kappa)+"kappa_"+str(self.lamb)+"lambda"
        else:
            self.name = name
        self.l = l
    def __call__(self,z):
        S = Variable(torch.zeros(z[:,0].shape))
        for i in range(self.n):
            tmp = Variable(torch.zeros(z[:,0].shape))
            for j in range(self.d):
                #print(z[:,self.hoppingTable[i][j*2]])
                tmp += z[:,self.hoppingTable[i][j*2]]
            #print("tmp: ",tmp)
            S += -2*self.kappa*tmp*z[:,i]
        S+=torch.sum(z**2,1)+self.lamb*torch.sum((z**2-1)**2,1)
        return S
    def createTable(self):
        for i in range(self.n):
            LK = self.n
            y = i
            hoppingTable.append([])
            for j in reversed(range(self.d)):
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

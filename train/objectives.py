import torch 
import numpy as np 

class Target(object):
    '''
    base class for target 
    '''
    def __init__(self,nvars,name = "Target"):
        self.nvars = nvars
        self.name = name

    def __call__(self, x):
        raise NotImplementedError(str(type(self)))

    def measure(self, x):
        return (x**2).sum(dim=1).numpy()

class Ring2D(Target):

    def __init__(self):
        super(Ring2D, self).__init__(2,'Ring2D')

    def __call__(self, x):
        return -(torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32

class Ring5(Target):

    def __init__(self):
        super(Ring5, self).__init__(2,'Ring5')

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


class Wave(Target):

    def __init__(self):
        super(Wave, self).__init__(2,'Wave')

    def __call__(self, x):
        w = torch.sin(np.pi*x[:, 0]/2.)
        return -0.5*((x[:, 1] -w)/0.4)**2

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
    def __call__(self,z):
        S = (torch.zeros(z[:,0].shape))
        for i in range(self.nvars):
            tmp = (torch.zeros(z[:,0].shape))
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

import torch
from torch.autograd import Variable
import numpy as np


class RealNVP(torch.nn.Module):

    def __init__(self, Nvars, Nlayers=2,Hs=10, Ht=10):
        """
        """
        super(RealNVP, self).__init__()
        self.Nvars = Nvars
        self.Nhalf = int(Nvars/2)
        self.Nlayers = Nlayers
        
        self.s = torch.nn.ModuleList() 
        self.t = torch.nn.ModuleList()
        for i in range(self.Nlayers):
            self.s.append ( 
                 torch.nn.Sequential(
                 torch.nn.Linear(self.Nhalf, Hs),
                 torch.nn.ReLU(),
                 torch.nn.Linear(Hs, Hs),
                 torch.nn.ReLU(),
                 torch.nn.Linear(Hs, self.Nhalf)
                 ))
                 
            self.t.append(
                 torch.nn.Sequential(
                 torch.nn.Linear(self.Nhalf, Ht),
                 torch.nn.ReLU(),
                 torch.nn.Linear(Ht, Ht),
                 torch.nn.ReLU(),
                 torch.nn.Linear(Ht, self.Nhalf)
                 ))

    def forward(self, x):
        """
        z = f(x)
        now only 2-layers 
        """

        y0 = x[:,0:self.Nhalf]
        y1 = x[:,self.Nhalf:self.Nvars] 

        self.logjac = Variable(torch.zeros(x.data.shape[0]))
        for i in range(self.Nlayers): 
            if (i%2==0):
                y1 = y1 * torch.exp(self.s[i](y0))  + self.t[i](y0)
                self.logjac += self.s[i](y0).sum(dim=1) 
            else:
                y0 = y0 * torch.exp(self.s[i](y1)) +  self.t[i](y1)
                self.logjac += self.s[i](y1).sum(dim=1)

        return torch.cat((y0, y1), 1)

    def backward(self, z):
        '''
        x = f^{-1}(z) = g(z)
        '''

        y0 = z[:,0:self.Nhalf]
        y1 = z[:,self.Nhalf:self.Nvars]
        
        for i in list(range(self.Nlayers))[::-1]:
            if (i%2==1):
                y0 = (y0 - self.t[i](y1)) * torch.exp(-self.s[i](y1))
            else:
                y1 = (y1 - self.t[i](y0)) * torch.exp(-self.s[i](y0))

        return torch.cat((y0, y1), 1)

    def logp(self, x):
        """
        logp(x) = logp(z) + sum_i s_i 
        """

        z = self.forward(x)
        return -0.5*(z**2).sum(dim=1) + self.logjac


if __name__=="__main__":
    import matplotlib.pyplot as plt 

    Nsamples = 1000
    Nvars = 2
    Nlayers = 4

    model = RealNVP(Nvars, Nlayers)
    print(model.parameters())
    
    x = Variable(torch.randn(Nsamples, Nvars))
    z = model.forward(x)

    #print (x)
    #print (z)
    #print (model.backward(z))

    print (x - model.backward(z)) # should be all zero 

    logp = model.logp(x)
    #print (logp)

    x = x.data.numpy()
    z = z.data.numpy()

    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')
    plt.scatter(z[:,0], z[:,1], alpha=0.5, label='$z$')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.ylabel('$x_1$')
    plt.xlabel('$x_2$')
    plt.legend() 

    plt.show()

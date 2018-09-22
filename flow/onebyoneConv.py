import torch
from torch import nn
import numpy as np
import scipy

from .flow import Flow

class OnebyoneConv(Flow):
    def __init__(self, h, w, c,prior = None, name = "OnebyoneConv"):
        super(OnebyoneConv,self).__init__(prior,name)
        t = np.random.randn(c,c)
        w,_ = np.linalg.qr(t)
        p,l,u = scipy.linalg.lu(w)

        self.p = nn.Parameter(torch.from_numpy(p).to(torch.float32),requires_grad=False)
        self.p_ = nn.Parameter(torch.from_numpy(p).to(torch.float32).inverse(),requires_grad=False)
        self.l = nn.Parameter(torch.from_numpy(l).to(torch.float32))
        self.u = nn.Parameter(torch.from_numpy(u).to(torch.float32))

    def inverse(self,y):
        s = torch.diagonal(self.u)
        inverseLogjac = torch.log(torch.abs(s)).sum()*y.shape[0]*y.shape[1]
        w  = torch.matmul(self.p,(torch.matmul(torch.tril(self.l),torch.triu(self.u))))
        y = torch.matmul(y.permute([0,2,3,1]),w.reshape(1,1,*w.shape)).permute(0,3,1,2)
        return y,inverseLogjac

    def forward(self,z):
        u_ = self.u.inverse()
        l_ = self.l.inverse()

        s = torch.diagonal(u_)
        forwardLogjac = torch.log(torch.abs(s)).sum()*z.shape[0]*z.shape[1]
        w_ = torch.matmul(torch.triu(u_),torch.matmul(torch.tril(l_),self.p_))
        z = torch.matmul(z.permute([0,2,3,1]),w_.reshape(1,1,*w_.shape)).permute(0,3,1,2)
        return z,forwardLogjac
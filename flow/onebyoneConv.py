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
        s = np.diag(u)
        sign = np.sign(s)
        logs = np.log(abs(s))
        u = np.triu(u,k=1)

        self.p = nn.Parameter(torch.from_numpy(p).to(torch.float32),requires_grad=False)
        self.p_ = nn.Parameter(torch.from_numpy(p).to(torch.float32).inverse(),requires_grad=False)
        self.l = nn.Parameter(torch.from_numpy(l).to(torch.float32))
        self.u = nn.Parameter(torch.from_numpy(u).to(torch.float32))
        self.sign = nn.Parameter(torch.from_numpy(sign).to(torch.float32),requires_grad=False)
        self.logs = nn.Parameter(torch.from_numpy(logs).to(torch.float32))

    def inverse(self,y):
        l = torch.tril(self.l,diagonal=-1)+torch.diag(torch.ones(self.l.shape[0])).to(self.p)
        u = torch.triu(self.u,diagonal=1)+torch.diag(self.sign*torch.exp(self.logs))
        w = torch.matmul(self.p,torch.matmul(l,u))
        inverseLogjac = self.logs.sum()*y.shape[-1]*y.shape[-2]*(torch.ones(y.shape[0]).to(self.p))
        yp = torch.matmul(y.permute([0,2,3,1]),w.reshape(1,1,*w.shape)).permute(0,3,1,2)
        return yp,inverseLogjac

    def forward(self,z):
        l = torch.tril(self.l,diagonal=-1)+torch.diag(torch.ones(self.l.shape[0])).to(self.p)
        u = torch.triu(self.u,diagonal=1)+torch.diag(self.sign*torch.exp(self.logs))

        u_ = torch.inverse(u)
        l_ = torch.inverse(l)
        w_ = torch.matmul(u_,torch.matmul(l_,self.p_))
        forwardLogjac = -self.logs.sum()*z.shape[-1]*z.shape[-2]*(torch.ones(z.shape[0]).to(self.p))
        zp = torch.matmul(z.permute([0,2,3,1]),w_.reshape(1,1,*w_.shape)).permute(0,3,1,2)
        return zp,forwardLogjac

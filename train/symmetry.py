import torch
from torch.nn import functional as F
import numpy as np
from random import randint
import math

from flow import Flow
from utils import logsumexp

class Symmetrized(Flow):
    def __init__(self,flow, symmetryList, name = None):
        super(Symmetrized,self).__init__(flow.prior,name)
        self.flow = flow
        self.symmetryList = symmetryList
        if name is None:
            name = "Symmetrized_"+flow.name
        self.name = name

    def sample(self, batchSize, prior=None):
        x,logp = self.flow.sample(batchSize,prior)
        noBatch = torch.LongTensor(batchSize).random_(0,len(self.symmetryList)+1)-1
        for i in range(len(self.symmetryList)):
            no = (noBatch==i)
            x[no] = self.symmetryList[i](x[no])
        return x,self.logProbability(x)

    def logProbability(self,x):
        logProbability = [self.flow.logProbability(x)]
        for op in self.symmetryList:
            logProbability.append(self.flow.logProbability(op(x)))
        logp = logsumexp(logProbability).view(-1)
        logp = logp - math.log(len(self.symmetryList)+1)
        return logp

    def inverse(self,z):
        batchSize = z.shape[0]
        x,_ = self.flow.inverse(z)
        noBatch = torch.LongTensor(batchSize).random_(0,len(self.symmetryList)+1)-1
        for i in range(len(self.symmetryList)):
            no = (noBatch==i)
            x[no] = self.symmetryList[i](x[no])
        return x,None

    def forward(self,x):
        xp = torch.zeros_like(x)
        logP = [self.flow.logProbability(x).reshape(1,-1)]
        logP += [self.flow.logProbability(self.symmetryList[i](x)).view(1,-1) for i in range(len(self.symmetryList))]
        logP = torch.cat(logP,0)
        logP = torch.nn.functional.softmax(logP,0)

        noBatch = torch.multinomial(logP.transpose(1,0),1).view(-1) - 1
        no = (noBatch == -1)
        xp[no] = x[no].detach()
        for i in range(len(self.symmetryList)):
            no = (noBatch == i)
            xp[no] = self.symmetryList[i](x[no]).detach()
        xp = xp.requires_grad_()
        z,_ = self.flow.forward(xp)
        return z,None

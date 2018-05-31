import torch
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
            name = "Symmetrized "+flow.name

    def sample(self, batchSize, prior=None):
        x,logp = self.flow.sample(batchSize,prior)
        for i in range(batchSize):
            no = randint(0,len(self.symmetryList))-1
            if no == -1:
                continue
            x[i] = self.symmetryList[no](x[i])
        return x

    def logProbability(self,x):
        logProbability = [self.flow.logProbability(x)]
        for op in self.symmetryList:
            logProbability.append(self.flow.logProbability(op(x)))
        logp = logsumexp(logProbability).view(-1)
        logp = logp - math.log(len(self.symmetryList)+1)
        return logp

    def generate(self,z):
        batchSize = z.shape[0]
        x,_ = self.flow.generate(z)
        for i in range(batchSize):
            no = randint(0,len(self.symmetryList))-1
            x[i] = self.symmetryList[no](x[i])
        return x,None

    def inference(self,x):
        z = torch.zeros_like(x)
        batchSize = x.shape[0]
        logP = [self.flow.logProbability(x).reshape(1,-1)]
        logP += [self.flow.logProbability(self.symmetryList[i](x)).view(1,-1) for i in range(len(self.symmetryList))]
        logP = torch.cat(logP,0)
        logP = torch.nn.functional.softmax(logP,1)

        no = torch.multinomial(logP.transpose(1,0),1).view(-1)
        for i in range(batchSize):
            if no[i] == 0:
                z[i],_ = self.flow.inference(x[i])
            else:
                z[i],_ = self.flow.inference(self.symmetryList[no[i].item()-1](x[i]))
        return z,None
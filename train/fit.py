import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC

class Buffer(object):
    def __init__(self,maximum,data=None):
        self.data = data
        self.maximum = maximum
    def draw(self,batchSize,testBatchSize=None):
        if batchSize >self.data.shape[0]:
            batchSize = self.data.shape[0]
        perm = torch.randperm(self.data.shape[0])
        if testBatchSize is None:
            return self.data[perm[:batchSize]]
        else:
            train =self.data[perm[:batchSize]]
            test = self.data[perm[batchSize:batchSize+testBatchSize]]
            return train,test
    def push(self,data):
        if self.data is None:
            self.data = data
        else:
            self.data = torch.cat([self.data,data],0)
        if self.data.shape[0] > self.maximum:
            self._maintain()
    def kill(self,ratio):
        pass
    def _maintain(self):
        perm = torch.randperm(self.data.shape[0])
        self.data = self.data[perm[:self.maximum]]

def train(model, Nepochs, supervised, buff, batchSize, modelname, lr = 5e-4,decay = 0.001,save = True, saveSteps=10, feed=True):
    LOSS=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.5, 0.9))
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

    for epoch in range(Nepochs):
        traindata = buff.draw(batchSize)
        x_data = Variable(traindata[:, 0:-1])

        if supervised:
            y_data = Variable(traindata[:, -1])

        logp,z = model.logProbabilityWithInference(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean()

        if feed:
            print ("epoch:",epoch, "loss:",loss.data[0], "z mean:",np.mean(z.data.numpy()),"z std:",np.std(z.data.numpy()))
        LOSS.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save and epoch%saveSteps==0:
            #saveD = {}
            #saveD["epoch"] = epoch
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

    return  x_data, model, LOSS

def test(model, supervised, buff, batchSize):

    testdata = buff.draw(batchSize)
    x_data = Variable(testdata[:, 0:-1])

    if supervised:
        y_data = Variable(testdata[:, -1])
        criterion = torch.nn.MSELoss(size_average=True)

    logp = model.logProbability(x_data)
    if supervised:
        loss = criterion(logp, y_data)
    else:
        loss = -logp.mean()

    print(torch.mean(loss))

    return loss
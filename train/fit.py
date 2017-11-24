import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC

def train(model, Nepochs, supervised, traindata, modelname, lr = 5e-4,decay = 0.001,save = True, saveSteps=10, feed=True):
    LOSS=[]

    x_data = Variable(traindata[:, 0:-1])

    if supervised:
        y_data = Variable(traindata[:, -1])
        criterion = torch.nn.MSELoss(size_average=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.5, 0.9))

    for epoch in range(Nepochs):

        logp = model.logProbability(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean()

        if feed:
            print (epoch, loss.data[0])
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

def test(model, supervised, testdata):
    x_data = Variable(traindata[:, 0:-1])

    if supervised:
        y_data = Variable(traindata[:, -1])
        criterion = torch.nn.MSELoss(size_average=True)

    logp = model.logProbability(x_data)
    if supervised:
        loss = criterion(logp, y_data)
    else:
        loss = -logp.mean()

    print(torch.mean(loss))

    return loss
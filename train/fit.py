import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC

def fit(model, Nepochs, supervised, traindata, modelname, ifCuda = False,double = True, lr = 0.01,decay = 0.001,save = True, saveSteps=10, feed=True):
    LOSS=[]
    
    x_data = Variable(traindata[:, 0:-1])

    if supervised:
        y_data = Variable(traindata[:, -1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

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

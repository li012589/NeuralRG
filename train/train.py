import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC

def fit(sList, tList, Nvars, Nepochs, supervised, traindata, modelname, ifCuda = False,double = True):
    LOSS=[]
    
    x_data = Variable(torch.from_numpy(traindata[:, 0:-1]))
    if ifCuda:
        x_data = x_data.cuda()

    if supervised:
        y_data = Variable(torch.from_numpy(traindata[:, -1]))
        if ifCuda:
            y_data = y_data.cuda()


    gaussian = Gaussian([Nvars])

    model = RealNVP([Nvars], sList, tList, gaussian, maskTpye="channel",name = modelname,double=double)

    if ifCuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

    for epoch in range(Nepochs):

        logp = model.logProbability(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean()

        print (epoch, loss.data[0])
        LOSS.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%10==0:
            #saveD = {}
            #saveD["epoch"] = epoch
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

    return Nvars, x_data, model, LOSS

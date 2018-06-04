import torch
import numpy as np
from utils import HMCwithAccept

def learn(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None, weight_decay = 0.001, adaptivelr = True, measureFn = None):
    if savePath is None:
        savePath = "./opt/"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ACC = []
    OBS = []


    for epoch in range(epochs):
        x,sampleLogProbability = flow.sample(batchSize)
        loss = sampleLogProbability.mean() - source.logProbability(x).mean()
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch, "L:",loss.item())

        LOSS.append([loss.item()])

        if save and epoch%saveSteps == 0:
            d = flow.save()
            torch.save(d,savePath+flow.name+".saving")

    return LOSS,ACC,OBS


def learnInterface(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None, weight_decay = 0.001, adaptivelr = True, HMCsteps = 10, HMCthermal = 10, HMCepsilon = 0.2, measureFn = None):

    def latentU(z):
        x,_ = flow.generate(z)
        return -(flow.prior.logProbability(z)+source.logProbability(x)-flow.logProbability(x))

    if savePath is None:
        savePath = "./opt/tmp"
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ZACC = []
    XACC = []
    ZOBS = []
    XOBS = []

    z_ = flow.prior.sample(batchSize)
    x_ = flow.prior.sample(batchSize)

    for epoch in range(epochs):
        x,sampleLogProbability = flow.sample(batchSize)
        loss = sampleLogProbability.mean() - source.logProbability(x).mean()
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch, "L:",loss.item())

        LOSS.append([loss.item()])

        if epoch%saveSteps == 0:
            if save:
                d = flow.save()
                torch.save(d,savePath+flow.name+".saving")
            z_,zaccept = HMCwithAccept(latentU,z_,HMCthermal,HMCsteps,HMCepsilon)
            x_,xaccept = HMCwithAccept(source.energy,x_,HMCthermal,HMCsteps,HMCepsilon)
            x_z,_ = flow.generate(z_)
            Zobs = measureFn(x_z)
            Xobs = measureFn(x_)
            print("accratio_z:",zaccept.mean().item(),"obs_z:",Zobs.mean(),  ' +/- ' , Zobs.std()/np.sqrt(1.*batchSize))
            print("accratio_x:",xaccept.mean().item(),"obs_x:",Xobs.mean(),  ' +/- ' , Xobs.std()/np.sqrt(1.*batchSize))

            ZACC.append(zaccept.mean().item())
            XACC.append(xaccept.mean().item())
            ZOBS.append(Zobs.mean())
            XOBS.append(Xobs.mean())

    return LOSS,ZACC,ZOBS,XACC,XOBS

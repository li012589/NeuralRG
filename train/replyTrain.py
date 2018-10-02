import torch
from torch import nn
import h5py
import numpy as np
import subprocess
import utils
from utils import HMCwithAccept
from .symmetry import Symmetrized
from torchvision.utils import make_grid, save_image

import flow
import source
import math
from flow import Flow


def replySymmetryMERAInit(L,d,nlayers,nmlp,nhidden,nrepeat,symmetryList,device,dtype,name = None, channel = 1, depthMERA = None):
    s = source.Gaussian([channel]+[L]*d)

    depth = int(math.log(L,2))*nrepeat*2

    coreSize = 4*channel

    MaskList = []
    for _ in range(depth):
        masklist = []
        for n in range(nlayers):
            if n%2 == 0:
                b = torch.zeros(1,coreSize)
                i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
                b.zero_()[:,i] = 1
                b=b.view(1,channel,2,2)
            else:
                b = 1-b
            masklist.append(b)
        masklist = torch.cat(masklist,0).to(torch.float32)
        MaskList.append(masklist)

    dimList = [coreSize]
    for _ in range(nmlp):
        dimList.append(nhidden)
    dimList.append(coreSize)

    layers = [flow.RNVP(MaskList[n], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(coreSize)]) for _ in range(nlayers)]) for n in range(depth)]

    f = flow.MERA(2,L,layers,nrepeat,depth = depthMERA,prior = s)
    #f = Symmetrized(f,symmetryList,name = name)
    f.to(device = device,dtype = dtype)
    return f

def learn(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None, weight_decay = 0.001, adaptivelr = False, measureFn = None):
    if savePath is None:
        savePath = "./opt/tmp/"
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
        x ,sampleLogProbability = flow.sample(batchSize)
        #loss = sampleLogProbability.mean() - source.logProbability(x).mean()
        lossorigin = (sampleLogProbability - source.logProbability(x))
        loss = lossorigin.mean()
        lossstd = lossorigin.std()
        del lossorigin
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch, "L:",loss.item(),"+/-",lossstd.item())

        LOSS.append([loss.item(),lossstd.item()])
        if adaptivelr:
            scheduler.step()
        if save and epoch%saveSteps == 0:
            d = flow.save()
            torch.save(d,savePath+flow.name+".saving")

    return LOSS,ACC,OBS


def replyLearnInterface(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10,savePath=None,keepSavings = 3, weight_decay = 0.001, adaptivelr = False, HMCsteps = 10, HMCthermal = 10, HMCepsilon = 0.2, measureFn = None,alpha=1.0):

    def cleanSaving(epoch):
        if epoch >= keepSavings*saveSteps:
            cmd =["rm","-rf",savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch-keepSavings*saveSteps)+".saving"]
            subprocess.check_call(cmd)
            cmd =["rm","-rf",savePath+"records/"+flow.name+"Record_epoch"+str(epoch-keepSavings*saveSteps)+".hdf5"]
            subprocess.check_call(cmd)

    def latentU(z):
        x,_ = flow.inverse(z)
        return -(flow.prior.logProbability(z)+source.logProbability(x)-flow.logProbability(x))

    if savePath is None:
        savePath = "./opt/tmp/"
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
        lossorigin = (sampleLogProbability - source.logProbability(x))
        lossstd = lossorigin.std()
        loss = (lossorigin.mean()+alpha*(sampleLogProbability.mean()-flow.logProbability(-x).mean()))
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        if adaptivelr:
            scheduler.step()

        del sampleLogProbability
        print("epoch:",epoch, "L:",loss.item(),"F:",lossorigin.mean().item(),"+/-",lossstd.item())
        del lossorigin

        LOSS.append([loss.item(),lossstd.item()])

        if (epoch%saveSteps == 0 and epoch > 50) or epoch == epochs:
            L = int(x.shape[-1]**0.5)
            configuration = torch.sigmoid(2.*x[:100])
            #img = make_grid(p, padding=1, nrow=10,normalize=False,scale_each=False).to(‘cpu’).numpy()
            save_image(configuration, savePath+'/proposals_{:04d}.png'.format(epoch), nrow=10, padding=1)
            #z_,zaccept = HMCwithAccept(latentU,z_.detach(),HMCthermal,HMCsteps,HMCepsilon)
            #x_,xaccept = HMCwithAccept(source.energy,x_.detach(),HMCthermal,HMCsteps,HMCepsilon)
            #with torch.no_grad():
                #x_z,_ = flow.inverse(z_)
                #z_last,_ = flow.forward(x_z)

            #with torch.no_grad():
                #Zobs = measureFn(x_z)
                #Xobs = measureFn(x_)
            print("Skip HMC")
            #print("accratio_z:",zaccept.mean().item(),"obs_z:",Zobs.mean(),  ' +/- ' , Zobs.std()/np.sqrt(1.*batchSize))
            #print("accratio_x:",xaccept.mean().item(),"obs_x:",Xobs.mean(),  ' +/- ' , Xobs.std()/np.sqrt(1.*batchSize))
            ZACC.append(np.nan)
            XACC.append(np.nan)
            ZOBS.append([np.nan,np.nan])
            XOBS.append([np.nan,np.nan])

            if save:
                with torch.no_grad():
                    samples,_ = flow.sample(batchSize)
                with h5py.File(savePath+"records/"+flow.name+"HMCresult_epoch"+str(epoch)+".hdf5","w") as f:
                    f.create_dataset("XZ",data=samples.detach().cpu().numpy())
                    f.create_dataset("Y",data=samples.detach().cpu().numpy())
                    f.create_dataset("X",data=samples.detach().cpu().numpy())

                #del x_z
                del samples

                with h5py.File(savePath+"records/"+flow.name+"Record_epoch"+str(epoch)+".hdf5", "w") as f:
                    f.create_dataset("LOSS",data=np.array(LOSS)[:,0])
                    f.create_dataset("LOSSSTD",data=np.array(LOSS)[:,1])
                    f.create_dataset("ZACC",data=np.array(ZACC))
                    f.create_dataset("ZOBS",data=np.array(ZOBS))
                    f.create_dataset("XACC",data=np.array(XACC))
                    f.create_dataset("XOBS",data=np.array(XOBS))
                d = flow.save()
                torch.save(d,savePath+"savings/"+flow.name+"Saving_epoch"+str(epoch)+".saving")
                cleanSaving(epoch)
        del x

    return LOSS,ZACC,ZOBS,XACC,XOBS

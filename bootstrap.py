import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np
import subprocess

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC, HMCSampler, fit

class Buffer(object):
    def __init__(self,maximum,data=None):
        self.data = data
        self.maximum = maximum
    def draw(self,batchSize):
        if batchSize >self.data.shape[0]:
            batchSize = self.data.shape[0]
        perm = np.random.permutation(self.data.shape[0])
        return self.data[perm[:batchSize]]
    def push(self,data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate([self.data,data],axis=0)
        if self.data.shape[0] > self.maximum:
            self._maintain()
    def kill(self,ratio):
        pass
    def _maintain(self):
        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm[:self.maximum]]

def boot(batchSize,Ntherm,Nsamples,Nskips,prior,target,sampler = MCMC):
    sampler = sampler(target, prior, collectdata=True)
    data,_,_ = sampler.run(batchSize, Ntherm, Nsamples, Nskips)
    return np.array(data)

def strap(model,nsteps,supervised,traindata,modelname,ifCuda,double,save,saveSteps):
    _,_,_ = fit(model,nsteps,supervised,traindata,modelname,ifCuda,double,save,saveSteps)

def test():
    pass

def main():
    #from utils.autoCorrelation import autoCorrelationTimewithErr
    #from utils.acceptRate import acceptanceRate

    l=6
    dims =3
    nvars = l**dims
    batchSize = 100
    trainSet = 500
    Ntherm = 300
    Nsamples = 500
    Nskips = 1
    kappa = 0.20
    lamb = 1.145
    maximum = 1000
    Nepochs = 3
    Nsteps = 500
    supervised = True
    saveSteps = 10
    testSteps = 100

    modelfolder = "data/bootstrap"
    cuda = False




    gaussian = Gaussian([nvars])
    buf = Buffer(maximum)
    target = Phi4(l, dims, kappa, lamb)
    
    #target = Ring2D()
    #nvars = 2

    sList = [MLP(nvars//2, 400),MLP(nvars//2, 400),MLP(nvars//2, 400),MLP(nvars//2, 400)]
    tList = [MLP(nvars//2, 400),MLP(nvars//2, 400),MLP(nvars//2, 400),MLP(nvars//2, 400)]

    gaussian = Gaussian([nvars])

    model = RealNVP([nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=True)
    if cuda:
        model = model.cuda()

    print("start initialization")

    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

    data = boot(batchSize,Ntherm,maximum//batchSize,Nskips,gaussian,target,sampler=HMCSampler)

    data = np.reshape(data,[-1,nvars+1])
    buf.push(data)
    print("finish initialise buffer")

    print("start bootstrap")

    for i in range(Nepochs):

        traindata = torch.from_numpy(buf.draw(trainSet))
        print(traindata.shape)
        strap(model,Nsteps,supervised,traindata,modelfolder,cuda,True,True,saveSteps)

        data = boot(batchSize,Ntherm,Nsamples,Nskips,model,target)
        data = np.reshape(data,[-1,nvars+1])
        buf.push(data)
        if i%testSteps == 0:
            test()


if __name__ == "__main__":
    main()


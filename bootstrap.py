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

def strap(model,nsteps,supervised,traindata,modelname,ifCuda,double,save=True,saveSteps=10):
    _,_,_ = fit(model,nsteps,supervised,traindata,modelname,ifCuda,double,save = save,saveSteps = saveSteps)

def test():
    pass

def main():
    #from utils.autoCorrelation import autoCorrelationTimewithErr
    #from utils.acceptRate import acceptanceRate
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-L",type=int,default=4,help="")
    parser.add_argument("-Dims",type=int,default=2,help="")
    parser.add_argument("-batchSize",type=int,default=16,help="")
    parser.add_argument("-trainSet",type=int,default=1000,help="")
    parser.add_argument("-Ntherm",type=int,default=300,help="")
    parser.add_argument("-Nsamples",type=int,default=500,help="")
    parser.add_argument("-Nskips",type=int,default=1,help="")
    parser.add_argument("-kappa",type=float,default=0.20,help="")
    parser.add_argument("-lamb",type=float,default=1.145,help="")
    parser.add_argument("-maximum",type=int,default=1000,help="")
    parser.add_argument("-Nepochs",type=int,default=100,help="")
    parser.add_argument("-Nsteps",type=int,default=500,help="")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true', help="supervised")
    group.add_argument("-unsupervised", action='store_true', help="unsupervised")

    parser.add_argument("-saveSteps",type=int,default=10,help="")
    parser.add_argument("-testSteps",type=int,default=100,help="")
    parser.add_argument("-Hs", type=int, default=10, help="")
    parser.add_argument("-Ht", type=int, default=10, help="")
    parser.add_argument("-float", action='store_true', help="use float32")
    parser.add_argument("-cuda", action='store_true', help="use GPU")
    args = parser.parse_args()
    double = not args.float
    nvars = args.L**args.Dims

    modelfolder = "data/bootstrap"




    gaussian = Gaussian([nvars])
    buf = Buffer(args.maximum)
    target = Phi4(args.L, args.Dims, args.kappa, args.lamb)
    
    #target = Ring2D()
    #nvars = 2

    sList = [MLP(nvars//2, args.Hs) for i in range(args.Nlayers)]
    tList = [MLP(nvars//2, args.Ht) for i in range(args.Nlayers)]

    gaussian = Gaussian([nvars])

    model = RealNVP([nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=double)
    if args.cuda:
        model = model.cuda()

    print("start initialization")

    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

    data = boot(args.batchSize,args.Ntherm,args.maximum//args.batchSize,args.Nskips,gaussian,target,sampler=HMCSampler)

    data = np.reshape(data,[-1,nvars+1])
    buf.push(data)
    print("finish initialise buffer")

    print("start bootstrap")

    for i in range(args.Nepochs):

        traindata = torch.from_numpy(buf.draw(args.trainSet))
        print(traindata.shape)
        strap(model,args.Nsteps,args.supervised,traindata,modelfolder,args.cuda,double)

        data = boot(args.batchSize,args.Ntherm,args.Nsamples,args.Nskips,model,target)
        data = np.reshape(data,[-1,nvars+1])
        #buf.push(data)
        if i%args.testSteps == 0:
            test()


if __name__ == "__main__":
    main()


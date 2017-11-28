import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np
import subprocess

from model import Gaussian,MLP,RealNVP,FC
from train import Ring2D, Ring5, Wave, Phi4, MCMC, HMCSampler, train, test, Buffer

def boot(batchSize,Ntherm,Nsamples,Nskips,prior,target,sampler = MCMC,double = True):
    sampler = sampler(target, prior, collectdata=True)
    data,_,_ = sampler.run(batchSize, Ntherm, Nsamples, Nskips)
    if double:
        return torch.Tensor(data).double()
    else:
        return torch.Tensor(data)

def strap(model,nsteps,supervised,buff,batchSize,modelname,ifCuda,double,save=True,saveSteps=10):
    _,_,_ = train(model,nsteps,supervised,buff,batchSize,modelname,save = save,saveSteps = saveSteps)

def check(target,model,Ntherm,Nsamples,supervised,buff,batchSize):
    loss = test(model,supervised,buff,batchSize)
    sampler = MCMC(target, model, collectdata=True)
    _,_,accratio = sampler.run(batchSize, Ntherm, Nsamples, 1)
    print(accratio)
    return loss,accratio

def main():
    #from utils.autoCorrelation import autoCorrelationTimewithErr
    #from utils.acceptRate import acceptanceRate
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-L",type=int,default=4,help="")
    parser.add_argument("-Dims",type=int,default=2,help="")
    parser.add_argument("-batchSize",type=int,default=16,help="")
    parser.add_argument("-trainSet",type=int,default=1000,help="")
    parser.add_argument("-testSet",type=int,default=500,help="")
    parser.add_argument("-Ntherm",type=int,default=300,help="")
    parser.add_argument("-Nsamples",type=int,default=500,help="")
    parser.add_argument("-Nskips",type=int,default=1,help="")
    parser.add_argument("-kappa",type=float,default=0.15,help="")
    parser.add_argument("-lamb",type=float,default=1.145,help="")
    parser.add_argument("-maximum",type=int,default=10000,help="")
    parser.add_argument("-Nepochs",type=int,default=100,help="")
    parser.add_argument("-Nsteps",type=int,default=500,help="")
    parser.add_argument("-testNtherm",type=int,default=300,help="")
    parser.add_argument("-testNsamples",type=int,default=1000,help="")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true', help="supervised")
    group.add_argument("-unsupervised", action='store_true', help="unsupervised")

    parser.add_argument("-saveSteps",type=int,default=10,help="")
    parser.add_argument("-testSteps",type=int,default=100,help="")
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-Hs", type=int, default=400, help="")
    parser.add_argument("-Ht", type=int, default=400, help="")
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

    #sList = [MLP(nvars//2, args.Hs) for i in range(args.Nlayers)]
    #tList = [MLP(nvars//2, args.Ht) for i in range(args.Nlayers)]

    sList = [FC([nvars//2, 100,200,300,200,100,nvars//2]) for _ in range(args.Nlayers)]
    tList = [FC([nvars//2, 100,200,300,200,100,nvars//2]) for _ in range(args.Nlayers)]

    gaussian = Gaussian([nvars])

    model = RealNVP([nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=double)
    if args.cuda:
        model = model.cuda()

    print(model)
    print("start initialization")

    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

    data = boot(args.batchSize,args.Ntherm,buf.capacity//args.batchSize,args.Nskips,gaussian,target,sampler=HMCSampler)

    data = data.view(-1,nvars+1)
    buf.push(data)
    print("finish initialise buffer")

    print("start bootstrap")

    for i in range(args.Nepochs):

        strap(model,args.Nsteps,args.supervised,buf, args.trainSet,modelfolder,args.cuda,double)

        _,accratio = check(target,model,args.testNtherm,args.testNsamples,args.supervised,buf,args.testSet)
        if(accratio>=0.25):
            data = boot(args.batchSize,args.Ntherm,args.Nsamples,args.Nskips,model,target)
        else:
            print("use hmc to generate some samples")
            torch.manual_seed(i*5+3)
            data = boot(args.batchSize,args.Ntherm,buf.capacity//args.batchSize,args.Nskips,gaussian,target,sampler=HMCSampler)
        data = data.view(-1,nvars+1)
        buf.push(data)


if __name__ == "__main__":
    main()


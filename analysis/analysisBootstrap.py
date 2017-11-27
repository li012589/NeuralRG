import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
import matplotlib.pyplot as plt 
import h5py
import argparse
import subprocess
from train import MCMC,HMCSampler
from torch.autograd import Variable
import numpy as np

from model import Gaussian, MLP, RealNVP
from train.objectives import Ring2D, Ring5, Wave, Phi4

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("-L",type=int,default=4,help="")
parser.add_argument("-Dims",type=int,default=2,help="")
parser.add_argument("-batchSize",type=int,default=16,help="")
parser.add_argument("-trainSet",type=int,default=1000,help="")
parser.add_argument("-Ntherm",type=int,default=300,help="")
parser.add_argument("-Nsamples",type=int,default=500,help="")
parser.add_argument("-Nskips",type=int,default=1,help="")
parser.add_argument("-kappa",type=float,default=0.15,help="")
parser.add_argument("-lamb",type=float,default=1.145,help="")
parser.add_argument("-Nlayers", type=int, default=8, help="")
parser.add_argument("-Hs", type=int, default=400, help="")
parser.add_argument("-Ht", type=int, default=400, help="")
parser.add_argument("-modelname", default=None, help="")
args = parser.parse_args()

nvars = args.L**args.Dims
target = Phi4(args.L, args.Dims, args.kappa, args.lamb)

modelfolder = "data/bootstrapTest"

sList = [MLP(nvars//2, args.Hs) for i in range(args.Nlayers)]
tList = [MLP(nvars//2, args.Ht) for i in range(args.Nlayers)]

gaussian = Gaussian([nvars])

z = gaussian.sample(2)

model = RealNVP([nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=True)

#print(model)

try:
    model.loadModel(torch.load(args.modelname))
    print('#load model', args.modelname)
except FileNotFoundError:
    print('model file not found:', args.modelname)
print("using model", args.modelname)

sampler = HMCSampler(target, gaussian, collectdata=True)

data,_,_ = sampler.run(args.batchSize, args.Ntherm, args.Nsamples, args.Nskips)
data = torch.Tensor(data).double()

data = data.view(-1,nvars+1)
data = data[:,:-1]

print(data.shape)

logp_model_train = model.logProbability(Variable(data))
logp_data_train = target(data)

print(type(logp_model_train))
print(type(logp_data_train))

plt.figure()
plt.scatter(logp_model_train.data.numpy(), logp_data_train.numpy(), alpha=0.5, label='training samples')
plt.xlabel('$\log{P(model)}$')
plt.ylabel('$\log{P(baseline)}$')
plt.legend()
plt.title("$\log{P(x)}$")
plt.show()
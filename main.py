import sys

import torch
from torch import nn
import numpy as np

import utils
import flow
import train
import source

from profilehooks import profile
import math
import h5py
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-folder", default='./opt/tmp/')

group = parser.add_argument_group('learning  parameters')
group.add_argument("-epochs", type=int, default=1000, help="")
group.add_argument("-batch", type=int, default=32, help="")
group.add_argument("-cuda", type=int, default=-1, help="use GPU")
group.add_argument("-double", action='store_true', help="use float64")
group.add_argument("-lr", type=float, default=0.001, help="learning rate")
group.add_argument("-save_period", type=int, default=10, help="")

group = parser.add_argument_group('network parameters')
group.add_argument("-load", action='store_true', help="if load from folder")
group.add_argument("-nlayers", type=int, default=4, help="# of layers in RNVP block")
group.add_argument("-nmlp",type = int, default=2,help="# of layers in MLP")
group.add_argument("-nhidden", type=int, default=32, help="")
group.add_argument("-nrepeat", type=int, default=2, help="repeat of mera block")

group = parser.add_argument_group('Ising target parameters')
#
group.add_argument("-L",type=int, default=4,help="linear size")
group.add_argument("-d",type=int, default=2,help="dimension")
group.add_argument("-T",type=float, default=2.269185314213022, help="Temperature")
group.add_argument("-fe_exact",type=float,default=None,help="fe_exact")
group.add_argument("-obs_exact",type=float,default=None,help="obs_exact")

args = parser.parse_args()

utils.createWorkSpace(args.folder)
if args.load:
    with h5py.File(args.folder+"/parameters.hdf5","r") as f:
        epochs = int(np.array(f["epochs"]))
        batch = int(np.array(f["batch"]))
        cuda = int(np.array(f["cuda"]))
        double = bool(np.array(f["double"]))
        lr = float(np.array(f["lr"]))
        save_period = int(np.array(f["save_period"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        T = float(np.array(f["T"]))
else:
    epochs = args.epochs
    batch = args.batch
    cuda = args.cuda
    double = args.double
    lr = args.lr
    save_period = args.save_period
    nlayers = args.nlayers
    nmlp = args.nmlp
    nhidden = args.nhidden
    nrepeat = args.nrepeat
    L = args.L
    d = args.d
    T = args.T
    fe_exact = args.fe_exact
    obs_exact = args.obs_exact
    with h5py.File(args.folder+"parameters.hdf5","w") as f:
        f.create_dataset("epochs",data=args.epochs)
        f.create_dataset("batch",data=args.batch)
        f.create_dataset("cuda",data=args.cuda)
        f.create_dataset("double",data=args.double)
        f.create_dataset("lr",data=args.lr)
        f.create_dataset("save_period",data=args.save_period)
        f.create_dataset("nlayers",data=args.nlayers)
        f.create_dataset("nmlp",data=args.nmlp)
        f.create_dataset("nhidden",data=args.nhidden)
        f.create_dataset("nrepeat",data=args.nrepeat)
        f.create_dataset("L",data=args.L)
        f.create_dataset("d",data=args.d)
        f.create_dataset("T",data=args.T)

device = torch.device("cpu" if cuda<0 else "cuda:"+str(cuda))

if double:
    dtype = torch.float64
else:
    dtype = torch.float32

s = source.Gaussian([L]*d)

d = source.Ising(L, d, T)
depth = int(math.log(L,2))*nrepeat*2

MaskList = []
for _ in range(depth):
    masklist = []
    for n in range(nlayers):
        if n%2 == 0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.view(1,2,2)
        else:
            b = 1-b
        masklist.append(b)
    masklist = torch.cat(masklist,0).to(torch.float32)
    MaskList.append(masklist)

dimList = [4]
for _ in range(nmlp):
    dimList.append(nhidden)
dimList.append(4)

layers = [flow.RNVP(MaskList[n], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[None]) for _ in range(nlayers)], [utils.SimpleMLPreshape(dimList,[nn.ELU() for _ in range(nmlp)]+[utils.ScalableTanh(4)]) for _ in range(nlayers)]) for n in range(depth)]

f = flow.MERA(2,L,layers,nrepeat,s)

def op(x):
    return -x

sym = [op]

f = train.Symmetrized(f,sym)
if args.load:
    import os
    import glob
    name = max(glob.iglob(args.folder+'savings/*.saving'), key=os.path.getctime)
    print("load saving at "+name)
    saved = torch.load(name)
    f.load(saved)

def measure(x):
        p = torch.sigmoid(2.*x).view(-1, d.nvars[0])
        s = 2.*p.data.cpu().numpy() - 1.
        sf = (s.mean(axis=1))**2 - (s**2).sum(axis=1)/d.nvars[0]**2  +1./d.nvars[0] #structure factor
        return  sf


LOSS,ZACC,ZOBS,XACC,XOBS = train.learnInterface(d,f,batch,epochs,save=True,savePath=args.folder,measureFn = measure)

import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from copy import deepcopy
import numpy as np
import math 
import matplotlib.pyplot as plt

from model import Gaussian, GMM, MLP,CNN,ResNet, RealNVP, ScalableTanh
from train import Ring2D, Ring5, Wave, Phi4, Mog2, Ising
from train import MCMC, Buffer
from hierarchy import MERA, MLPreshape 
from learn_realnvp import learn_acc # FIXME 

if __name__=="__main__":
    import h5py
    import subprocess
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-folder", default='data/learn_mera/',
                    help="where to store results")

    group = parser.add_argument_group('learning  parameters')
    group.add_argument("-Nepochs", type=int, default=500, help="")
    group.add_argument("-Batchsize", type=int, default=64, help="")
    group.add_argument("-cuda", action='store_true', help="use GPU")
    group.add_argument("-double", action='store_true', help="use float64")

    group.add_argument("-lr", type=float, default=0.001, help="learning rate")
    group.add_argument("-epsilon", type=float, default=1.0, help="acceptance term")
    #group.add_argument("-alpha", type=float, default=0.0, help="sjd term")
    group.add_argument("-beta", type=float, default=1.0, help="temperature term")
    #group.add_argument("-gamma", type=float, default=0.0, help="weight to the mse loss")
    group.add_argument("-delta", type=float, default=0.0, help="weight to the nll loss on data")
    group.add_argument("-omega", type=float, default=0.0, help="weight to the KL(model|data)")

    group = parser.add_argument_group('network parameters')
    group.add_argument("-modelname", default=None, help="load model")
    group.add_argument("-prior", default='gaussian', help="prior distribution")
    group.add_argument("-Nlayers", type=int, default=8, help="# of layers in RNVP block")
    group.add_argument("-Ndisentangler", type=int, default=1, help="# of disentanglers")
    group.add_argument("-Hs", type=int, default=10, help="")
    group.add_argument("-Ht", type=int, default=10, help="")
    group.add_argument("-train_model", action='store_true', help="actually train model")
    group.add_argument("-train_prior", action='store_true', help="if we train the prior")

    group = parser.add_argument_group('mc parameters')
    group.add_argument("-Ntherm", type=int, default=10, help="")
    group.add_argument("-Nsteps", type=int, default=10, help="steps used in training")
    group.add_argument("-Nskips", type=int, default=10, help="")
    group.add_argument("-Nsamples", type=int, default=10, help="")

    group = parser.add_argument_group('target parameters')
    group.add_argument("-target", default='ring2d', help="target distribution")
    #Mog2 
    group.add_argument("-offset",type=float, default=2.0,help="offset of mog2")
    #Ising
    group.add_argument("-L",type=int, default=2,help="linear size")
    group.add_argument("-d",type=int, default=1,help="dimension")
    group.add_argument("-T",type=float, default=2.6, help="Temperature")
    group.add_argument("-exact",type=float,default=None,help="exact")

    args = parser.parse_args()
    cuda = None
    if args.cuda:
        cuda = 0

    if args.double:
        print ('use float64')
    else:
        print ('use float32')

    if args.target == 'ring2d':
        target = Ring2D()
    elif args.target == 'ring5':
        target = Ring5()
    elif args.target == 'wave':
        target = Wave()
    elif args.target == 'mog2':
        target = Mog2(args.offset)
    elif args.target == 'phi4':
        target = Phi4(4,2,0.15,1.145)
    elif args.target == 'ising':
        target = Ising(args.L, args.d, args.T, cuda, args.double)
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    Nvars = target.nvars 

    if args.prior == 'gaussian':
        if args.d== 2:
            prior = Gaussian([args.L, args.L], requires_grad = args.train_prior)
        else:
            prior = Gaussian([Nvars], requires_grad = args.train_prior)
    elif args.prior == 'gmm':
        prior = GMM([Nvars])
    else:
        print ('what prior?', args.prior)
        sys.exit(1)
    print ('prior:', prior.name)

    key = args.folder \
          + args.target 

    if (args.target=='ising'):
        key += '_L' + str(args.L)\
              + '_d' + str(args.d) \
              + '_T' + str(args.T)

    key+=  '_Nl' + str(args.Nlayers) \
          + '_Hs' + str(args.Hs) \
          + '_Ht' + str(args.Ht) \
          + '_epsilon' + str(args.epsilon) \
          + '_beta' + str(args.beta) \
          + '_delta' + str(args.delta) \
          + '_omega' + str(args.omega) \
          + '_Batchsize' + str(args.Batchsize) \
          + '_Ntherm' + str(args.Ntherm) \
          + '_Nsteps' + str(args.Nsteps) \
          + '_Nskips' + str(args.Nskips) \
          + '_lr' + str(args.lr) 

    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)
    
    #RNVP block
    kernel_size = [2]*args.d 
    mlpsize = int(np.product(np.array(kernel_size)))
    depth = int(math.log(Nvars,mlpsize))*(args.Ndisentangler +1)

    print ('depth of the mera network', depth)
    sList = [[MLPreshape(mlpsize, args.Hs, activation=ScalableTanh([mlpsize])) for _ in range(args.Nlayers)] for l in range(depth)]
    tList = [[MLPreshape(mlpsize, args.Ht) for _ in range(args.Nlayers)] for l in range(depth)]
    masktypelist = ['channel', 'channel'] * (args.Nlayers//2)
    
    #assamble RNVP blocks into a MERA
    layers = [RealNVP(kernel_size, 
                      sList[l], 
                      tList[l], 
                      None, 
                      masktypelist) for l in range(depth)] 
    
    model = MERA(args.d, kernel_size, Nvars, layers, prior, metaDepth =args.Ndisentangler+1, name=key)

    if args.modelname is not None:
        try:
            model.loadModel(torch.load(args.modelname))
            print('#load model', args.modelname)
        except FileNotFoundError:
            print('model file not found:', args.modelname)

    if args.cuda:
        model = model.cuda()
        print("moving model to GPU")

    if args.train_model:
        print("train model", key)
        model, LOSS = learn_acc(target, model, args.Nepochs,args.Batchsize, 
                                args.Ntherm, args.Nsteps, args.Nskips,
                                epsilon=args.epsilon,beta=args.beta, 
                                delta=args.delta, omega=args.omega, lr=args.lr, 
                                cuda = cuda, exact=args.exact)

    sampler = MCMC(target, model, collectdata=True)
    
    samples, proposals, measurements, accratio, _, _ = sampler.run(args.Batchsize, args.Ntherm, args.Nsamples, args.Nskips, cuda = cuda)
    
    h5filename = key + '_mc.h5'
    print("save at: " + h5filename)
    h5 = h5py.File(h5filename, 'w')
    params = h5.create_group('params')
    params.create_dataset("Nvars", data=target.nvars)
    params.create_dataset("Nlayers", data=args.Nlayers)
    params.create_dataset("Hs", data=args.Hs)
    params.create_dataset("Ht", data=args.Ht)
    params.create_dataset("target", data=args.target)
    params.create_dataset("model", data=model.name)
    results = h5.create_group('results')
    results.create_dataset("obs", data=np.array(measurements))
    results.create_dataset("samples", data=samples.cpu().numpy())
    results.create_dataset("proposals", data=proposals.cpu().numpy())
    if args.train_model:
        results.create_dataset("loss", data=np.array(LOSS))
    h5.close()
    print ('#accratio:', accratio)

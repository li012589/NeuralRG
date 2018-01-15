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

import argparse
import h5py

parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('model parameters')
group.add_argument("-modelname", default=None, help="load model")
group.add_argument("-h5file", default = None, help = "parameters saving file")
args = parser.parse_args()

h5 = h5py.File(args.h5file,"r")
L = int(np.array(h5['params']['Ls']))
Nvars = int(np.array(h5['params']['Nvarss']))
d = int(np.array(h5['params']['ds']))
Hs = int(np.array(h5['params']['Hss']))
Ht = int(np.array(h5['params']['Hts']))
Ndisentangler = int(np.array(h5['params']['Ndisentanglers']))
Nlayers = int(np.array(h5['params']['Nlayerss']))

kernel_size = [2]*d
mlpsize = int(np.product(np.array(kernel_size)))
nperdepth = (Ndisentangler +1) # number of disentangers + number of decimator at each RG step
depth = int(math.log(Nvars,mlpsize))

print ('depth of the mera network', depth)
sList = [[MLPreshape(mlpsize, Hs, activation=ScalableTanh([mlpsize]))
          for _ in range(Nlayers)]
         for l in range(nperdepth*depth)]

tList = [[MLPreshape(mlpsize, Ht) for _ in range(Nlayers)]
                                           for l in range(nperdepth*depth)]

masktypelist = ['channel', 'channel'] * (Nlayers//2)
prior = Gaussian([L,L])
#assamble RNVP blocks into a MERA
layers = [RealNVP(kernel_size,
                  sList[l],
                  tList[l],
                  None,
                  masktypelist) for l in range(nperdepth*depth)]

model = MERA(d, kernel_size, Nvars, layers, prior, metaDepth =Ndisentangler+1)

model.loadModel(torch.load(args.modelname))

z = prior(1)

x = model.generate(z,save=True)

N = len(model.saving)//(Ndisentangler+1)

for i in range(N):
    print (model.saving[(i-1)*(Ndisentangler+1)+Ndisentangler])
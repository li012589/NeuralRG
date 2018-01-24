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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm 
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
import h5py

parser = argparse.ArgumentParser(description='')

parser.add_argument("-modelname", default=None, help="load model")
parser.add_argument("-h5file", default = None, help = "parameters saving file")
parser.add_argument("-scale", default =0, type=int, help = "scale")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")
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

depth = int(math.log(Nvars,kernel_size[0]**2))
sidLen = int(math.sqrt(Nvars))
masks = []
for n in range(1,depth):
    tmp = np.zeros([kernel_size[0]**n,kernel_size[0]**n])
    tmp[0,0]=1
    tmp = np.tile(tmp,(sidLen//kernel_size[0]**n,sidLen//kernel_size[0]**n))
    masks.append((torch.from_numpy(tmp).byte()))

pos = {}
cols ={}
row = 0
for i,mask in enumerate(masks):
    tmp = torch.nonzero(mask)
    pos[i] = tmp
    num = int(np.sqrt(tmp.size()[0]))
    cols[i] = num
    row += num
row += L
cols[i+1]=L

z_prior = prior(1)
fig = plt.figure(figsize=(8, 8))
todraw = []
for ix in range(L):
    for iy in range(L):
        z = deepcopy(z_prior)
        z[0, ix, iy] = -z[0,ix,iy]
        #print (z)

        x = model.generate(z,save=True)

        N = len(model.saving)//(Ndisentangler+1)

        data = model.saving[args.scale*(Ndisentangler+1)+Ndisentangler].data

        todraw.append(data[0])

l = int(np.sqrt(data.numpy().size))

todraw = torch.cat(todraw).view(L,L,-1)

tmp = 0
for i in range(len(pos)):
    for j,p in enumerate(pos[i]):
        if j >= cols[i]:
            j = j-cols[i] + L
        plt.subplot(row,L,tmp+j+1)
        plt.imshow(todraw[p[0]][p[1]].view(l,l),cmap=cm.gray)
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    tmp+=cols[i]*L

for j,p in enumerate(todraw.view(-1,l,l)):
    plt.subplot(row,L,tmp+j+1)
    plt.imshow(p.view(l,l),cmap=cm.gray)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)

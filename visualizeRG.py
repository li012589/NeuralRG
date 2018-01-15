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

parser.add_argument("-modelname", default=None, help="load model")
parser.add_argument("-h5file", default = None, help = "parameters saving file")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-show", action='store_true',  help="show figure right now")
group.add_argument("-outname", default="result.pdf",  help="output pdf file")
args = parser.parse_args()
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
print (x)
N = len(model.saving)//(Ndisentangler+1)

import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

labels = ["(a)", "(b)", "(c)", "(d)"]
fig = plt.figure(figsize=(8, 5))
for i in range(N):
    
    plt.subplot(1,N,i+1)
    data = model.saving[i*(Ndisentangler+1)+Ndisentangler].data.numpy()
    
    L = int(np.sqrt(data.size))
    data.shape = (L, L)

    im = plt.imshow(data, cmap=cm.gray)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation='horizontal')

    at = AnchoredText(labels[i],prop=dict(size=18), frameon=False,loc=2, bbox_to_anchor=(-0.1, 1.3), bbox_transform=ax.transAxes,)
    plt.gca().add_artist(at)

    print (model.saving[i*(Ndisentangler+1)+Ndisentangler])


if args.show:
    plt.show()
else:
    plt.savefig(args.outname, dpi=300, transparent=True)

import os
import sys
import h5py
import argparse
import subprocess
from train import MCMC,HMCSampler
import torch
from torch.autograd import Variable
import numpy as np

from model import Gaussian, MLP, RealNVP
from train.objectives import Ring2D, Ring5, Wave, Phi4

parser = argparse.ArgumentParser(description='')
parser.add_argument("-target", default='ring2d',
                    help="target distribution")
parser.add_argument("-collectdata", action='store_true',
                    help="collect data")
parser.add_argument("-folder", default='data/',
                    help="where to store results")
parser.add_argument("-savename", default=None, help="")

group = parser.add_argument_group('mc parameters')
group.add_argument("-sampler",default='metropolis',help="")
group.add_argument("-Batchsize", type=int, default=16, help="")
group.add_argument("-Ntherm", type=int, default=300, help="")
group.add_argument("-Nsamples", type=int, default=1000, help="")
group.add_argument("-Nskips", type=int, default=1, help="")

group = parser.add_argument_group('network parameters')
group.add_argument("-modelname", default=None, help="")
group.add_argument("-Nlayers", type=int, default=8, help="")
group.add_argument("-Hs", type=int, default=10, help="")
group.add_argument("-Ht", type=int, default=10, help="")
group.add_argument("-cuda",action='store_true',help='move model to GPU')
args = parser.parse_args()

if args.target == 'ring2d':
    target = Ring2D()
elif args.target == 'ring5':
    target = Ring5()
elif args.target == 'wave':
    target = Wave()
elif args.target == 'phi4':
    target = Phi4(3, 2, 1.0, 1.0)
else:
    print('what target ?', args.target)
    sys.exit(1)

gaussian = Gaussian([target.nvars])

if args.modelname is None:
    model = gaussian
    print("using gaussian")
else:
    sList = [MLP(target.nvars // 2, args.Hs) for _ in range(args.Nlayers)]
    tList = [MLP(target.nvars // 2, args.Ht) for _ in range(args.Nlayers)]

    model = RealNVP([target.nvars], sList, tList, gaussian, name=None)
    try:
        model.loadModel(torch.load(args.modelname))
        print('#load model', args.modelname)
    except FileNotFoundError:
        print('model file not found:', args.modelname)
    print("using model", args.modelname)
    if args.cuda:
        model = model.cuda()
        print("moving model to GPU")
if args.sampler == 'metropolis':
    print("using MCMC as sampler")
    sampler = MCMC(target, model, collectdata=args.collectdata)
elif args.sampler == 'hmc':
    print("using HMC as sampler")
    sampler = HMCSampler(target, model, collectdata=args.collectdata)
data,measurements,accratio = sampler.run(args.Batchsize, args.Ntherm, args.Nsamples, args.Nskips)
cmd = ['mkdir', '-p', args.folder]
subprocess.check_call(cmd)
if args.savename is None:
    key = args.folder \
          + args.target \
          + '_Nl' + str(args.Nlayers) \
          + '_Hs' + str(args.Hs) \
          + '_Ht' + str(args.Ht)
else:
    key = args.savename
if not args.collectdata:
    key += "_nosample"
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
results.create_dataset("accratio", data= accratio)
if args.collectdata:
    results.create_dataset("samples", data=data)
h5.close()
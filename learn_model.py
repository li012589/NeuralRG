import os
import sys
sys.path.append(os.getcwd())
import torch
#torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, train, Buffer

if __name__=="__main__":
    import h5py
    import subprocess
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-Hs", type=int, default=10, help="")
    parser.add_argument("-Ht", type=int, default=10, help="")
    parser.add_argument("-Nepochs", type=int, default=500, help="")
    parser.add_argument("-target", default='ring2d', help="target distribution")
    parser.add_argument("-cuda", action='store_true', help="use GPU")
    parser.add_argument("-float", action='store_true', help="use float32")
    parser.add_argument("-folder", default='data/', help="where to store results")
    parser.add_argument("-traindata", default=None, help="train data")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true', help="supervised")
    group.add_argument("-unsupervised", action='store_true', help="unsupervised")
    args = parser.parse_args()

    if args.target == 'ring2d':
        target = Ring2D()
    elif args.target == 'ring5':
        target = Ring5()
    elif args.target == 'wave':
        target = Wave()
    elif args.target == 'phi4':
        target = Phi4(4,2,0.15,1.145)
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    sl_or_ul = '_sl' if args.supervised else '_ul'
    modelfolder = args.traindata.replace('_mc.h5', sl_or_ul+'/')
    h5filename = args.traindata.replace('_mc', sl_or_ul)

    print (modelfolder)
    print (h5filename)

    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

    h5 = h5py.File(args.traindata,'r')
    if not args.float:
        xy = np.array(h5['results']['samples'],dtype=np.float64)
    else:
        xy = np.array(h5['results']['samples'],dtype=np.float32)
    h5.close()

    Nvars = xy.shape[-1] -1
    xy.shape = (-1, Nvars +1)
    xy = torch.from_numpy(xy)
    if args.cuda:
        xy = xy.cuda()

    buf = Buffer(int(xy.shape[0]),xy)

    sList = [MLP(Nvars//2, args.Hs) for i in range(args.Nlayers)]
    tList = [MLP(Nvars//2, args.Ht) for i in range(args.Nlayers)]

    gaussian = Gaussian([Nvars])

    model = RealNVP([Nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=not args.float)
    if args.cuda:
        model = model.cuda()

    x_data, model, LOSS= train(model,
                               args.Nepochs,
                               args.supervised,
                               buf,
                               int(xy.shape[0]),
                               modelfolder)
    #after training, generate some data from the network
    Ntest = 1000
    if args.cuda:
        z = model.prior(Ntest, volatile=True).cuda()# prior
    else:
        z = model.prior(Ntest, volatile=True)# prior

    if args.float:
        z=z.float()

    x = model.generate(z)

    # on training data
    logp_model_train = model.logProbability(x_data)
    logp_data_train = target(x_data.data)

    # on test data
    logp_model_test = model.logProbability(x)
    logp_data_test = target(x.data)

    h5 = h5py.File(h5filename,'w')
    params = h5.create_group('params')
    params.create_dataset("Nvars", data=Nvars)
    params.create_dataset("Nlayers", data=args.Nlayers)
    params.create_dataset("Hs", data=args.Hs)
    params.create_dataset("Ht", data=args.Nlayers)
    params.create_dataset("target", data=args.target)
    params.create_dataset("supervised", data=args.supervised)
    params.create_dataset("unsupervised", data=args.unsupervised)

    results = h5.create_group('results')
    results.create_dataset("train_data",data=x_data.cpu().data.numpy())
    results.create_dataset("generated_data",data=x.cpu().data.numpy())
    results.create_dataset("logp_model_train",data=logp_model_train.cpu().data.numpy())
    results.create_dataset("logp_model_test",data=logp_model_test.cpu().data.numpy())
    results.create_dataset("logp_data_train",data=logp_data_train.cpu().numpy())
    results.create_dataset("logp_data_test",data=logp_data_test.cpu().numpy())
    results.create_dataset("loss",data=LOSS)

    h5.close()

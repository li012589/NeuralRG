import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC

def learn_acc(target, model, Nepochs, Batchsize, Nsamples, modelname, lr =1e-3, weight_decay = 0.001,save = True, saveSteps=10):
    LOSS=[]

    sampler = MCMC(target, model, collectdata=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(Nepochs):
        samples, _ ,accratio,res = sampler.run(Batchsize, 0, Nsamples, 1)

        #print (accratio, type(accratio)) 
        loss = -res.mean()

        print ("epoch:",epoch, "loss:",loss.data[0], "acc:", accratio)
        LOSS.append([loss.data[0], accratio])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save and epoch%saveSteps==0:
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

            samples = np.array(samples)
            samples.shape = (Batchsize*Nsamples, -1)
            x = model.sample(1000)
            x = x.cpu().data.numpy()
  
            plt.figure()
            plt.scatter(x[:,0], x[:,1], alpha=0.5, label='proposals')
            plt.scatter(samples[:,0], samples[:,1], alpha=0.5, label='samples')
            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend()
            plt.savefig(model.name+'/epoch%g.png'%(epoch)) 
            plt.close() 

    return model, LOSS

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
    parser.add_argument("-Batchsize", type=int, default=64, help="")
    parser.add_argument("-Nsamples", type=int, default=100, help="")
    parser.add_argument("-cuda", action='store_true', help="use GPU")
    parser.add_argument("-float", action='store_true', help="use float32")

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

    modelfolder = 'data/learn_acc'
    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

    Nvars = 2 

    sList = [MLP(Nvars//2, args.Hs) for i in range(args.Nlayers)]
    tList = [MLP(Nvars//2, args.Ht) for i in range(args.Nlayers)]

    gaussian = Gaussian([Nvars])

    model = RealNVP([Nvars], sList, tList, gaussian, maskTpye="channel",name = modelfolder,double=not args.float)
    if args.cuda:
        model = model.cuda()

    model, LOSS= learn_acc(target, model, args.Nepochs,args.Batchsize, args.Nsamples,'learn_acc')

    import matplotlib.pyplot as plt 
    plt.figure()
    LOSS = np.array(LOSS)
    plt.subplot(211)
    plt.plot(LOSS[:, 0], label='loss')
    plt.subplot(212)
    plt.plot(LOSS[:, 1], label='acc')
    plt.xlabel('iterations')

    plt.show()

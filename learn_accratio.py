import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian,MLP,RealNVP
from train import Ring2D, Ring5, Wave, Phi4, MCMC


def learn_acc(target, model, Nepochs, Batchsize, modelname, lr = 5e-4,decay = 0.001,save = True, saveSteps=10):
    LOSS=[]

    sampler = MCMC(target, model, collectdata=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.5, 0.9))

    for epoch in range(Nepochs):
        samples, _ ,accratio = sampler.run(Batchsize, 100, 10, 10)

        #print (accratio, type(accratio)) 
        #loss = -torch.log(accratio.mean())
        loss = -accratio.mean()

        print ("epoch:",epoch, "loss:",loss.data[0])
        LOSS.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save and epoch%saveSteps==0:
            #saveD = {}
            #saveD["epoch"] = epoch
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

    return samples, model, LOSS

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

    samples, model, LOSS= learn_acc(target, model,args.Nepochs,args.Batchsize, 'learn_acc')

    #after training, generate some data from the network
    Ntest = 1000
    if args.cuda:
        z = model.prior(Ntest, volatile=True).cuda()# prior
    else:
        z = model.prior(Ntest, volatile=True)# prior

    if args.float:
        z=z.float()

    x = model.generate(z)
    x = x.cpu().data.numpy()
    samples = np.array(samples)
    
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='proposals')
    plt.scatter(samples[:,0], samples[:,1], alpha=0.5, label='samples')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

    plt.show()

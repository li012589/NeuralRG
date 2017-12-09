import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import Gaussian, Cauchy, MLP,RealNVP, ScalableTanh
from train import Ring2D, Ring5, Wave, Phi4, Mog2, Ising
from train import MCMC, Buffer 

class Offset(torch.nn.Module):
    '''
    offset a scalar 
    '''
    def __init__(self):
        super(Offset, self).__init__()
        self.offset = torch.nn.Parameter(torch.DoubleTensor([0]))    
    def forward(self, x):
        return x + self.offset

def learn_acc(target, model, Nepochs, Batchsize, Nsteps, Nskips, modelname, alpha=0.0, beta=1.0, gamma=0.0, lr =1e-3, weight_decay = 0.001,save = True, saveSteps=10):
    LOSS=[]

    sampler = MCMC(target, model, collectdata=True)
    
    offset = Offset()
    buff = Buffer(10000)

    if (gamma>0):
        params = list(model.parameters()) + list(offset.parameters())
    else:
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    Nanneal = Nepochs//2
    dbeta = (1.-beta)/Nanneal
    
    plt.ion() 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    l1, = ax1.plot([], [],'o', alpha=0.5, label='proposals')
    l2, = ax1.plot([], [],'*', alpha=0.5, label='samples')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    fig1.canvas.draw()


    fig2 = plt.figure()
    ax21 = fig2.add_subplot(211)
    l3, = ax21.plot([], [], label='loss')
    ax21.legend()

    ax22 = fig2.add_subplot(212, sharex=ax21)
    l4, = ax22.plot([], [], label='acc')
    ax22.set_xlim([0, Nepochs])
    ax22.legend()
    plt.xlabel('epochs')
    fig2.canvas.draw()

    for epoch in range(Nepochs):
        samples, proposals ,_, accratio, res, sjd = sampler.run(Batchsize, 0, Nsteps, Nskips)

        ######################################################
        #mes loss on the proposals
        xy = np.array(proposals)
        xy.shape = (Batchsize*Nsteps, -1)
        xy = torch.from_numpy(xy)
        buff.push(xy)

        traindata = buff.draw(Batchsize)
        x_data = Variable(traindata[:, :-1])
        y_data = Variable(traindata[:, -1])
        y_pred = model.logProbability(x_data)
        mse = (offset(y_pred) - y_data).pow(2)
        ######################################################

        loss = -res.mean() - alpha * sjd.mean() + gamma * mse.mean()

        alpha *= 0.98 
        if (epoch < Nanneal):
            beta += dbeta
        sampler.set_beta(beta)
        
        print ("epoch:",epoch, "loss:",loss.data[0], "acc:", accratio, "beta:", beta, "offset:", offset.offset.data[0])
        LOSS.append([loss.data[0], accratio])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save and epoch%saveSteps==0:
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

            samples = np.array(samples)
            samples.shape = (Batchsize*Nsteps, -1)

            proposals = np.array(proposals)
            proposals.shape = (Batchsize*Nsteps, -1)
            
            l1.set_xdata(proposals[:,0])
            l1.set_ydata(proposals[:,1])

            l2.set_xdata(samples[:,0])
            l2.set_ydata(samples[:,1])
            ax1.set_title('epoch=%g'%(epoch))

            fig1.canvas.draw()
            fig1.savefig(model.name+'/epoch%g.png'%(epoch)) 

            loss4plot = np.array(LOSS)
        
            l3.set_xdata(range(len(LOSS)))
            l4.set_xdata(range(len(LOSS)))
            l3.set_ydata(loss4plot[:,0])
            l4.set_ydata(loss4plot[:,1])
            ax21.relim()
            ax21.autoscale_view() 
            ax22.relim()
            ax22.autoscale_view() 

            fig2.canvas.draw()
            plt.pause(0.001)

    fig2.savefig(model.name + '/loss.png')
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
    parser.add_argument("-prior", default='gaussian', help="prior distribution")
    parser.add_argument("-Batchsize", type=int, default=64, help="")
    parser.add_argument("-cuda", action='store_true', help="use GPU")
    parser.add_argument("-float", action='store_true', help="use float32")

    parser.add_argument("-alpha", type=float, default=0.0, help="sjd term")
    parser.add_argument("-beta", type=float, default=1.0, help="temperature term")
    parser.add_argument("-gamma", type=float, default=0.0, help="weight to the mse loss")
    parser.add_argument("-folder", default='data/',
                    help="where to store results")

    group = parser.add_argument_group('mc parameters')
    group.add_argument("-Ntherm", type=int, default=100, help="")
    group.add_argument("-Nsamples", type=int, default=100, help="")
    group.add_argument("-Nsteps", type=int, default=10, help="steps used in training")
    group.add_argument("-Nskips", type=int, default=10, help="")

    group = parser.add_argument_group('target parameters')
    #Mog2 
    group.add_argument("-offset",type=float, default=2.0,help="offset of mog2")
    #Ising
    group.add_argument("-L",type=int, default=2,help="linear size")
    group.add_argument("-d",type=int, default=1,help="dimension")
    group.add_argument("-K",type=float, default=1.0,help="K")

    args = parser.parse_args()

    modelfolder = 'data/learn_acc'
    cmd = ['mkdir', '-p', modelfolder]
    subprocess.check_call(cmd)

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
        target = Ising(args.L, args.d, args.K)
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    Nvars = target.nvars 

    if args.prior == 'gaussian':
        prior = Gaussian([Nvars])
    elif args.prior == 'cauchy':
        prior = Cauchy([Nvars])
    else:
        print ('what prior?', args.prior)
        sys.exit(1)


    sList = [MLP(Nvars//2, args.Hs, ScalableTanh(Nvars//2)) for i in range(args.Nlayers)]
    tList = [MLP(Nvars//2, args.Ht, F.linear) for i in range(args.Nlayers)] 

    model = RealNVP([Nvars], sList, tList, prior, maskTpye="channel",name = modelfolder,double=not args.float)
    if args.cuda:
        model = model.cuda()

    model, LOSS = learn_acc(target, model, args.Nepochs,args.Batchsize, 
                            args.Nsteps, args.Nskips,
                            'learn_acc', alpha=args.alpha, beta=args.beta, gamma =args.gamma)

    sampler = MCMC(target, model, collectdata=True)
    _, _, measurements, _, _, _= sampler.run(args.Batchsize, args.Ntherm, args.Nsamples, args.Nskips)

    cmd = ['mkdir', '-p', args.folder]
    subprocess.check_call(cmd)
    key = args.folder \
          + args.target \
          + '_Nl' + str(args.Nlayers) \
          + '_Hs' + str(args.Hs) \
          + '_Ht' + str(args.Ht)
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
    results.create_dataset("loss", data=np.array(LOSS))
    h5.close()



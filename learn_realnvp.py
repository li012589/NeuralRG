import os
import sys
sys.path.append(os.getcwd())
import torch
torch.manual_seed(42)
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import numpy as np
import matplotlib.pyplot as plt

from model import Gaussian, GMM, MLP,CNN,ResNet, RealNVP, ScalableTanh
from train import Ring2D, Ring5, Wave, Phi4, Mog2, Ising
from train import MCMC, Buffer
from copy import deepcopy

def learn_acc(target, model, Nepochs, Batchsize, Ntherm, Nsteps, Nskips, shape, 
              epsilon = 1.0, beta=1.0, delta=0.0, omega=0.0, 
              lr =1e-3, weight_decay = 0.001, save = True, saveSteps=10, cuda = None, 
              exact= None):

    LOSS = []
    OBS = []

    sampler = MCMC(target, model, collectdata=True)
    
    #offset = Offset()
    #if cuda is not None:
    #    offset = offset.cuda(cuda)
    #buff_proposals = Buffer(10000)
    buff_samples = Buffer(10*Batchsize)

    params = list(model.parameters()) 
    #if (gamma>0):
    #    params += list(offset.parameters())
    
    #filter out those we do not want to train
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    Nanneal = Nepochs//2
    dbeta = (1.-beta)/Nanneal
    
    plt.ion() 

    #samples 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    l1, = ax1.plot([], [],'o', alpha=0.5, label='proposals')
    l2, = ax1.plot([], [],'*', alpha=0.5, label='samples')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    fig1.canvas.draw()

    #loss, acceptance, observable 
    fig2 = plt.figure(figsize=(8, 8))
    ax21 = fig2.add_subplot(311)
    plt.title('$\epsilon=%g, \delta=%g, \omega=%g$'%(epsilon, delta, omega))
    l3, = ax21.plot([], [], label='loss')
    ax21.legend()

    ax22 = fig2.add_subplot(312, sharex=ax21)
    l4, = ax22.plot([], [], label='acc')
    ax22.set_xlim([0, Nepochs])
    ax22.legend()

    ax23 = fig2.add_subplot(313, sharex=ax21)
    l5, = ax23.plot([], [], label='obs')
    if exact is not None:
        ax23.axhline(exact, color='r')
    ax23.set_xlim([0, Nepochs])
    ax23.legend()

    plt.xlabel('epochs')
    fig2.canvas.draw()
    
    #parameter hist  
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    [n, X, V]=ax3.hist([], bins=50)
    fig3.canvas.draw()

    for epoch in range(Nepochs):

        for epoch in range(Nepochs):
            if (buff_samples.maximum > Batchsize):
                # draw starting state from the sampler buffer
                zinit = buff_samples.draw(Batchsize)[:, :-1].contiguous().view(-1, *shape)
        else:
            zinit = None

        samples, proposals, measurements, accratio, res, kld  = sampler.run(Batchsize, 
                                                                            Ntherm, 
                                                                            Nsteps, 
                                                                            Nskips,
                                                                            zinit, 
                                                                            cuda=cuda)

        ######################################################
        #push samples to buffer
        xy = samples.view(Batchsize*Nsteps,-1)
        #data argumentation using invertion symmetry
        xy_invert = deepcopy(xy)
        xy_invert[:, :-1] = -xy_invert[:, :-1]
        xy = torch.stack([xy, xy_invert],0).view(Batchsize*Nsteps*2,-1)
        buff_samples.push(xy)
        ######################################################


        loss = kld.mean() 

        if (epoch < Nanneal):
            beta += dbeta
        target.set_beta(beta)
        
        print ("epoch:",epoch
               ,"loss:",loss.data[0]
               ,"acc:", accratio
               ,"beta:", beta
               #,"offset:", offset.offset.data[0]
               ,"obs", np.array(measurements).mean()
               #"mu", model.prior.mu1.data[0], model.prior.mu2.data[0]
               )

        LOSS.append([loss.data[0], accratio])
        OBS.append(np.array(measurements).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step(loss.data[0])

        if save and epoch%saveSteps==0:
            saveDict = model.saveModel({})
            torch.save(saveDict, model.name+'/epoch'+str(epoch))

            samples = samples.cpu().numpy()
            samples.shape = (Batchsize*Nsteps, -1)

            proposals = proposals.cpu().numpy()
            proposals.shape = (Batchsize*Nsteps, -1)
            
            l1.set_xdata(proposals[:,0])
            l1.set_ydata(proposals[:,1])

            l2.set_xdata(samples[:,0])
            l2.set_ydata(samples[:,1])
            ax1.set_title('epoch=%g'%(epoch))

            ax1.relim()
            ax1.autoscale_view() 

            fig1.canvas.draw()
            fig1.savefig(model.name+'/epoch%g.png'%(epoch)) 

            loss4plot = np.array(LOSS)
            obs4plot = np.array(OBS)
        
            l3.set_xdata(range(len(LOSS)))
            l4.set_xdata(range(len(LOSS)))
            l5.set_xdata(range(len(OBS)))
            l3.set_ydata(loss4plot[:,0])
            l4.set_ydata(loss4plot[:,1])
            l5.set_ydata(obs4plot)
            ax21.relim()
            ax21.autoscale_view() 
            ax22.relim()
            ax22.autoscale_view() 
            ax23.relim()
            ax23.autoscale_view() 

            fig2.canvas.draw()

            paramslist = []
            for p in params:
                paramslist += list(p.data.cpu().numpy().ravel()) # could be sppeed up
            ax3.cla()
            [n,X,V]=ax3.hist(paramslist)
            ax3.relim()
            ax3.autoscale_view() 
            fig3.canvas.draw()

            plt.pause(0.001)

    fig2.savefig(model.name + '/loss.png')
    return model, LOSS

if __name__=="__main__":
    import h5py
    import subprocess
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-folder", default='data/learn_acc/',
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
    group.add_argument("-Nlayers", type=int, default=8, help="")
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

    netDimension = 2
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
        netDimension = args.d+1
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    Nvars = target.nvars 

    if args.prior == 'gaussian':
        if netDimension == 3:
            prior = Gaussian([1, args.L, args.L], requires_grad = args.train_prior)
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
    
    #MLP
    if netDimension == 2:
        input_size= [Nvars]
        sList = [MLP(Nvars, args.Hs) for i in range(args.Nlayers)]
        tList = [MLP(Nvars, args.Ht) for i in range(args.Nlayers)]

    else:

        input_size= [1, args.L, args.L]
        #CNN 
        snet = [[args.Hs,3,1,1],
                [args.Hs,1,1,0],
                [1,3,1,1]]

        tnet = [[args.Ht,3,1,1],
                [args.Ht,1,1,0],
                [1,3,1,1]]

        sList = [CNN(snet, activation=ScalableTanh(input_size)) for i in range(args.Nlayers)]
        tList = [CNN(tnet) for i in range(args.Nlayers)]
   
        #Resnet 
        #sList = [ResNet(args.Hs, activation=ScalableTanh(input_size)) for i in range(args.Nlayers)]
        #tList = [ResNet(args.Ht) for i in range(args.Nlayers)]
        #masktypelist = ['checkerboard0', 'checkerboard1', 
        #                'leftright0', 'leftright1',
        #                'updown0', 'updown1',
        #                'bars0', 'bars1',
        #                'stripes0', 'stripes1'
        #                ]
        #Resnet
    if netDimension == 2:
        masktypelist = ['evenodd', 'evenodd'] * (args.Nlayers//2)
    else:
        masktypelist = ['checkerboard', 'checkerboard'] * (args.Nlayers//2)

    model = RealNVP(input_size, sList, tList, prior, 
                    masktypelist, name = key)
    if args.double:
        model = model.double()
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
                                args.Ntherm, args.Nsteps, args.Nskips,input_size,
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

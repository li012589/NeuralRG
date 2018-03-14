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

from train import MCMC, Buffer
from copy import deepcopy

def MClearn(target, model, Nepochs, Batchsize, Ntherm, Nsteps, Nskips, shape,
            delta=0.0, omega=0.0, lr =1e-3, weight_decay = 0.001, save = True,
            saveSteps=10, cuda = None, exact= None):

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

    plt.ion() 

    #samples 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    l1, = ax1.plot([], [],'o', alpha=0.5, label='proposals')
    l2, = ax1.plot([], [],'*', alpha=0.5, label='samples')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend(loc='upper left')
    fig1.canvas.draw()

    #loss, acceptance, observable 
    fig2 = plt.figure(figsize=(8, 8))
    ax21 = fig2.add_subplot(411)
    plt.title('$\delta=%g, \omega=%g$'%(delta, omega))
    l3, = ax21.plot([], [], label='loss')
    ax21.legend()

    ax22 = fig2.add_subplot(412, sharex=ax21)
    l3half, = ax22.plot([], [], label='$\mathcal{L}$')
    ax22.set_xlim([0, Nepochs])
    ax22.legend()

    ax23 = fig2.add_subplot(413, sharex=ax21)
    l4, = ax23.plot([], [], label='acc')
    ax23.set_xlim([0, Nepochs])
    ax23.legend()

    ax24 = fig2.add_subplot(414, sharex=ax21)
    l5, = ax24.plot([], [], label='obs')
    if exact is not None:
        ax24.axhline(exact, color='r')
    ax24.set_xlim([0, Nepochs])
    ax24.legend()

    plt.xlabel('epochs')
    fig2.canvas.draw()
    
    #parameter hist  
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    [n, X, V]=ax3.hist([], bins=50)
    fig3.canvas.draw()

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

        x_data = Variable(buff_samples.draw(Batchsize)[:, :-1].contiguous().view(-1,*shape))
        #nll loss on the samples
        nll_samples = -model.logProbability(x_data)
        ######################################################

        loss = delta*nll_samples.mean()  + omega * kld.mean()

        print ("epoch:",epoch
               ,"loss:",loss.data[0], -res.mean().data[0], nll_samples.mean().data[0], kld.mean().data[0]
               ,"acc:", accratio
               ,"obs", np.array(measurements).mean()
               )

        LOSS.append([loss.data[0], kld.mean().data[0], accratio])
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
            l3half.set_xdata(range(len(LOSS)))
            l4.set_xdata(range(len(LOSS)))
            l5.set_xdata(range(len(OBS)))

            l3.set_ydata(loss4plot[:,0])
            l3half.set_ydata(loss4plot[:,1])
            l4.set_ydata(loss4plot[:,2])
            l5.set_ydata(obs4plot)

            ax21.relim()
            ax21.autoscale_view() 
            ax22.relim()
            ax22.autoscale_view() 
            ax23.relim()
            ax23.autoscale_view() 
            ax24.relim()
            ax24.autoscale_view() 

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

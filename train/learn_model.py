if __name__ =="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model import Gaussian,MLP,RealNVP
from train.objectives import ring2d as target_logp 

def fit(Nlayers, Hs, Ht, Nepochs, supervised):
    xy = np.loadtxt('train.dat', dtype=np.float32)
    x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
    print (x_data.data.shape)

    if supervised:
        y_data = Variable(torch.from_numpy(xy[:, -1]))
        print (y_data.data.shape)

    print (x_data.data.shape)

    Nvars = x_data.data.shape[-1]
    print (Nvars)

    #model = RealNVP(Nvars, Nlayers=Nlayers, Hs=Hs, Ht=Ht)
    gaussian = Gaussian([Nvars])

    sList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]
    tList = [MLP(2, 10), MLP(2, 10), MLP(2, 10), MLP(2, 10)]

    model = RealNVP([Nvars], sList, tList, gaussian)
    model.createMask(10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

    for epoch in range(Nepochs):

        logp = model.logProbability(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean() # ?? not very clear why

        print (epoch, loss.data[0]) 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

    saveDict = realNVP.saveModel({})
    torch.save(saveDict, './'+model.name)
    return Nvars, x_data, model

def visualize(Nvars, x_data, model):

    #after training, generate some data from the network
    Nsamples = 1000 # test samples 
    z = Variable(torch.randn(Nsamples, Nvars), volatile=True)
    x = model.generate(z)  

    # on training data 
    logp_model_train = model.logProbability(x_data)
    logp_data_train = target_logp(x_data)

    # on test data
    logp_model_test = model.logProbability(x)
    logp_data_test = target_logp(x)

    plt.figure() 
    plt.scatter(logp_model_train.data.numpy(), logp_data_train.data.numpy(), alpha=0.5, label='train')
    plt.scatter(logp_model_test.data.numpy(), logp_data_test.data.numpy(), alpha=0.5, label='test')

    plt.xlabel('model')
    plt.ylabel('baseline')

    plt.legend() 
    #plt.show() 
    #import sys
    #sys.exit(0)
    
    x_data = x_data.data.numpy()
    x = x.data.numpy()
    #overwites training data 
    #f = open('train.dat','w')
    #for i in range(x.shape[0]):
    #    f.write("%g %g %g\n"%(x[i, 0], x[i, 1], test_logprob(x[i, :])))  
    #f.close() 

    plt.figure()
    plt.scatter(x_data[:,0], x_data[:,1], alpha=0.5, label='original')
    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='generated')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend() 

    ###########################
    #x = np.arange(-5, 5, 0.01)
    #y = np.arange(-5, 5, 0.01)
    #X, Y = np.meshgrid(x, y)
    #Z = np.zeros_like(X)
    #for i in range(Z.shape[0]):
    #    for j in range(Z.shape[1]):
    #        Z[i,j] = np.exp(test_logprob([X[i,j], Y[i,j]]) ) 
    #plt.contour(X, Y, Z)
    ###########################

    plt.show()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nvars", type=int, default=2, help="")
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-Hs", type=int, default=10, help="")
    parser.add_argument("-Ht", type=int, default=10, help="")
    parser.add_argument("-Nepochs", type=int, default=500, help="")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true', help="supervised")
    group.add_argument("-unsupervised", action='store_true', help="unsupervised")
    args = parser.parse_args()
    
    Nvars, x_data, model = fit(args.Nlayers, 
                               args.Hs, 
                               args.Ht, 
                               args.Nepochs, 
                               args.supervised)

    visualize(Nvars, x_data, model) 


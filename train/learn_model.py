import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model.realnvp import RealNVP 
from train.generate_samples import test_logprob 

def fit(supervised):
    xy = np.loadtxt('train.dat', dtype=np.float32)
    x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
    print (x_data.data.shape)

    if supervised:
        y_data = Variable(torch.from_numpy(xy[:, -1]))
        print (y_data.data.shape)

    print (x_data.data.shape)

    Nvars = x_data.data.shape[-1]
    print (Nvars)

    model = RealNVP(Nvars, Nlayers=8, Hs=10, Ht=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

    for epoch in range(500):

        logp = model.logp(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean() # NLL 

        print (epoch, loss.data[0]) 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
 
    return Nvars, x_data, model

def visualize(Nvars, x_data, model):

    #after training, generate some data from the network
    Nsamples = 1000 # test samples 
    z = Variable(torch.randn(Nsamples, Nvars), volatile=True)
    x = model.backward(z)

    # on training data 
    logp_model_train = model.logp(x_data)
    logp_data_train = [test_logprob(x_data[i].data.numpy()) for i in range(x_data.data.shape[0]) ] 

    # on test data
    logp_model_test = model.logp(x)
    logp_data_test = [test_logprob(x[i].data.numpy()) for i in range(x.data.shape[0])]

    plt.figure() 
    plt.scatter(logp_model_train.data.numpy(), logp_data_train, alpha=0.5, label='train')
    plt.scatter(logp_model_test.data.numpy(), logp_data_test, alpha=0.5, label='test')

    plt.legend() 
    #plt.show() 
    #import sys
    #sys.exit(0)

    x = x.data.numpy()

    plt.figure()
    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.ylabel('$x_1$')
    plt.xlabel('$x_2$')
    plt.legend() 

    ###########################
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = np.exp( test_logprob([X[i,j], Y[i,j]]) ) 
    plt.contour(X, Y, Z)
    ###########################

    plt.show()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true',  help="supervised")
    group.add_argument("-unsupervised", action='store_true',  help="unsupervised")
 
    args = parser.parse_args()
    
    Nvars, x_data, model = fit(args.supervised)
    visualize(Nvars, x_data, model) 


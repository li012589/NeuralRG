import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model.realnvp import RealNVP 
from train.generate_samples import test_logprob 

def inference(model):

    #after training, generate some data from the network
    Nsamples = 1000 # test samples 
    z = Variable(torch.randn(Nsamples, model.Nvars), volatile=True)
    x = model.backward(z)

    x = x.data.numpy()

    plt.figure()
    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
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

    parser.add_argument("-Nvars", type=int,  help="")
    parser.add_argument("-Nlayers", type=int,  help="")
    parser.add_argument("-Hs", type=int,  help="")
    parser.add_argument("-Ht", type=int,  help="")
    args = parser.parse_args()

    model = RealNVP(Nvars = args.Nvars, 
                    Nlayers = args.Nlayers, 
                    Hs = args.Hs, 
                    Ht = args.Ht)

    model.load_state_dict(torch.load(model.name))

    inference(model) 

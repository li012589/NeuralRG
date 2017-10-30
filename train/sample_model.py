import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model.realnvp import RealNVP 
from train.objectives import ring2d as target_logp 

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
    #plot contour of the target potential 
    grid = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)

    x = np.array( [[X[i,j], Y[i, j]] for j in range(Z.shape[1]) for i in range(Z.shape[0])])
    logp = target_logp(torch.from_numpy(x))
    counter = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = np.exp ( logp[counter] ) 
            counter += 1
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

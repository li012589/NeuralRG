import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model import Gaussian,MLP,RealNVP
from train.objectives import ring2d as target_logp 

def inference(model):

    #after training, generate some data from the network
    Ntest = 1000 # test samples 
    z = model.prior(Ntest)#Variable(torch.randn(self.batchsize, self.nvars), volatile=True) # prior 
    x = model.generate(z)

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

    parser.add_argument("-Nvars", type=int, default=2, help="")
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-Hs", type=int, default=10, help="")
    parser.add_argument("-Ht", type=int, default=10, help="")
 
    args = parser.parse_args()

    sList = [MLP(args.Nvars//2, args.Hs) for _ in range(args.Nlayers)] 
    tList = [MLP(args.Nvars//2, args.Ht) for _ in range(args.Nlayers)] 

    gaussian = Gaussian([args.Nvars])

    model = RealNVP([args.Nvars], sList, tList, Gaussian([args.Nvars]))
    model.loadModel(torch.load(model.name))
 

    inference(model) 

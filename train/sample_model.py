import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 
import matplotlib.pyplot as plt 

from model import Gaussian,MLP,RealNVP
from train.objectives import Ring2D, Ring5 

def inference(model, target):

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
    logp = target(torch.from_numpy(x))
    counter = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = np.exp ( logp[counter] ) 
            counter += 1
    plt.contour(X, Y, Z)
    ###########################

    plt.show()

if __name__=="__main__":
    import sys, os 
    import argparse
    import re 
    import h5py 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-modelname", default=None, help="model name")
    args = parser.parse_args()
        
    h5filename = re.search('(.*)_[su]l',args.modelname).group(1)
    h5filename += '_mc.h5'

    h5 = h5py.File(h5filename,'r')
    Nvars = int(h5['params']['Nvars'][()])
    Nlayers = int(h5['params']['Nlayers'][()])
    Hs = int(h5['params']['Hs'][()])
    Ht = int(h5['params']['Ht'][()])
    targetname = h5['params']['target'][()]
    h5.close() 

    print (Nvars, Nlayers, Hs, Ht)

    sList = [MLP(Nvars//2, Hs) for _ in range(Nlayers)] 
    tList = [MLP(Nvars//2, Ht) for _ in range(Nlayers)] 

    gaussian = Gaussian([Nvars])

    model = RealNVP([Nvars], sList, tList, gaussian, args.modelname)
    model.loadModel(torch.load(model.name))

    if targetname == 'ring2d':
        target = Ring2D()
    elif targetname == 'ring5':
        target = Ring5()
    elif targetname == 'wave':
        target = Wave()
    else:
        print ('what target ?', targetname)
        sys.exit(1) 
 
    inference(model, target) 

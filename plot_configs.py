from numpy import loadtxt, sqrt 
import matplotlib.pyplot as plt 
import argparse
import os 
from matplotlib import cm
import h5py 
import numpy as np 

parser = argparse.ArgumentParser(description='')
parser.add_argument("-filename", help="filename")
args = parser.parse_args()

h5 = h5py.File(args.filename,'r')
proposals = np.array(h5['results']['proposals'])
samples = np.array(h5['results']['samples'])
h5.close()
 
files = []

sigmoid = lambda x: 1./(1.+np.exp(-x))
for i in range(proposals.shape[0]):

    left = proposals[i, 1, :-1]
    right = samples[i, 1, :-1]
    #left = sigmoid(2.* proposals[i, 1, :-1])
    #right = sigmoid(2.* samples[i, 1, :-1])
    N = len(left)
    L = int(sqrt(N))

    #left = (left>np.random.rand(N))
    #right = (right>np.random.rand(N))

    left.shape = (L, L)
    right.shape = (L, L)

    print (i, N, L)
    plt.cla()
    plt.subplot(121)
    plt.imshow(left, interpolation='nearest', cmap=cm.gray)
    plt.subplot(122)
    plt.imshow(right, interpolation='nearest', cmap=cm.gray)
    plt.title('$step=%g$'%(i))

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    fname = '_tmp%04d.png'%(i)
    plt.savefig(fname)
    files.append(fname)

os.system('convert -trim -delay 20 -dispose previous _tmp*.png animation.gif')

#for fname in files:
#    os.remove(fname)

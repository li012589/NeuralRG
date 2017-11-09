import numpy as np 
import matplotlib.pyplot as plt
import os 

import argparse 

parser = argparse.ArgumentParser(description='')
parser.add_argument("-filename", default='train.dat', help="filename")
args = parser.parse_args()

data = np.loadtxt(args.filename, usecols=(0, 1))

for i in range(len(data)):
    plt.scatter(data[:i, 0], data[:i, 1], alpha=0.5, label='training samples', color='C0')

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.savefig('_tmp%04d.png'%(i))

gifname = args.filename.replace('.dat', '.gif')
os.system('convert -trim -delay 20 -dispose previous _tmp*.png %s'%(gifname))

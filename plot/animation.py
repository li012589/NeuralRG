import numpy as np 
import matplotlib.pyplot as plt
import os 

train = np.loadtxt('train.dat', usecols=(0, 1))

for i in range(len(train)):
    plt.scatter(train[:i, 0], train[:i, 1], alpha=0.5, label='training samples', color='C0')

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig('_tmp%04d.png'%(i))

os.system('convert -trim -delay 20 -dispose previous _tmp*.png animation.gif')
os.remove('_tmp*.png')

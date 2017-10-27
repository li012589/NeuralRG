import numpy as np
from scipy.linalg import eigh

__all__ = ["test_logprob", "transform"]

A = np.array([[1.,  0.5], 
              [0.5, 0.5]])

def test_logprob(x):
    '''
    unnormalized logprob
    '''
    return -0.5*np.dot(np.transpose(x), np.dot(A, x)) 

def transform(z):
    '''
    how to obtain p(x) distribution using direct sampling
    '''

    w, v = eigh(A)
    z /= np.sqrt(w)
    return np.dot(v, z)

if __name__=='__main__':
    Nsamples = 10000 
    z = np.random.randn(Nsamples, 2)

    print('#x, logp(x)')
    for i in range(Nsamples):
        #randomly generate some train sample (not according to their logp at the moment)
        #print (z[i, 0], z[i, 1], test_logprob(z[i]))  

        x = transform(z[i])
        print (x[0], x[1], test_logprob(x))  

#import matplotlib.pyplot as plt 
#x = np.loadtxt('train.dat', dtype=np.float32, usecols=(0, 1))
#plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')
#plt.show()

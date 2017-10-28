import numpy as np
np.random.seed(42)
from scipy.linalg import eigh

__all__ = ["test_logprob", "transform"]

A = np.array([[1.,  0.5], 
              [0.5, 0.5]])

def test_logprob(x):
    '''
    unnormalized logprob
    '''
    return -0.5*np.dot(np.transpose(x), np.dot(A, x)) 
    #return -(np.sqrt(x[0]**2 + x[1]**2)-2.0)**2/0.32

def transform(z):
    '''
    how to obtain p(x) distribution using direct sampling
    '''

    w, v = eigh(A)
    z /= np.sqrt(w)
    return np.dot(v, z)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nsamples", type=int, default=10000,help="number of training samples")
    parser.add_argument("-filename", default='train.dat',  help="filename")
    parser.add_argument("-plot", action='store_true',  help="plot data")
 
    args = parser.parse_args()
 
    z = np.random.rand(args.Nsamples, 2)
    z = 8*(z-0.5)

    f = open(args.filename,'w')
    for i in range(args.Nsamples):
        #randomly generate some train sample (not according to their logp at the moment)
        #print (z[i, 0], z[i, 1], test_logprob(z[i]))  
        f.write("%g %g %g\n"%(z[i, 0], z[i, 1], test_logprob(z[i])))  
    f.close() 

        #x = transform(z[i])
        #print (x[0], x[1], test_logprob(x))  
    
    if args.plot:
        import matplotlib.pyplot as plt 
        x = np.loadtxt(args.filename, dtype=np.float32, usecols=(0, 1))
        plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')
        plt.show()

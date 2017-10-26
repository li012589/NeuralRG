import numpy as np

def test_logprob(x):
    '''
    unnormalized logprob
    '''
    x0, x1 = x
    return -x0**2/2.- x1**2/4.-x0*x1/2.

if __name__=='__main__':
    #randomly generate some train sample (not according to their logp at the moment)
    Nsamples = 5000 
    x = np.random.randn(Nsamples, 2)

    print('#x, logp(x)')
    for i in range(Nsamples):
        print (x[i, 0], x[i, 1], test_logprob(x[i]))  

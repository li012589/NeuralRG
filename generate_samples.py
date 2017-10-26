import numpy as np

def test_logprob(x):
    '''
    a test logprob
    '''
    x0, x1 = x
    return -x0**2/2.- x1**2/4.-x0*x1/2.

#randomly generate some train sample (not according to their logp at the moment)
Nsamples = 2000 
x = np.random.randn(Nsamples, 2)

print('#x, logp(x)')
for i in range(Nsamples):
    print (x[i, 0], x[i, 1], test_logprob(x[i]))  


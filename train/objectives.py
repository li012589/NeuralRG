import torch 
from torch.autograd import Variable 


def ring2d(x):
    '''
    unnormalized logprob
    '''
    return -(torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32



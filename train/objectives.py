import torch 

def ring2d(x):
    '''
    unnormalized logprob
    '''
    return -(torch.sqrt((x**2).sum(dim=1))-2.0)**2/0.32


def ring5(x):

    u1 = (torch.sqrt((x**2).sum(dim=1)) - 1.) **2 /0.04
    u2 = (torch.sqrt((x**2).sum(dim=1)) - 2.) **2 /0.04
    u3 = (torch.sqrt((x**2).sum(dim=1)) - 3.) **2 /0.04
    u4 = (torch.sqrt((x**2).sum(dim=1)) - 4.) **2 /0.04
    u5 = (torch.sqrt((x**2).sum(dim=1)) - 5.) **2 /0.04

    u1 = u1.view(-1, 1)
    u2 = u2.view(-1, 1)
    u3 = u3.view(-1, 1)
    u4 = u4.view(-1, 1)
    u5 = u5.view(-1, 1)

    u = torch.cat((u1, u2, u3, u4, u5), dim=1)
    return -torch.min(u, dim=1)[0]


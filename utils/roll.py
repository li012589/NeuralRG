import torch

def roll(x, step, axis):
    shape = x.shape
    for i,s in enumerate(step):
        if s >=0:
            x1 = x.narrow(axis[i],0,s)
            x2 = x.narrow(axis[i],s,shape[axis[i]]-s)
        else:
            x2 = x.narrow(axis[i],shape[axis[i]]+s,-s)
            x1 = x.narrow(axis[i],0,shape[axis[i]]+s)
        x = torch.cat([x2,x1],axis[i])
    return x

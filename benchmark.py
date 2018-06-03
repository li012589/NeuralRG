import utils
import numpy as np
import flow
import torch
from torch import nn
import train
import source
from profilehooks import profile

@profile
def traint():
    return train.learn(target,m,batchSize,epochs,save = False)

p = source.Gaussian([4,4])

BigList = []
for _ in range(2*2*2):
    maskList = []
    for n in range(4):
        if n %2==0:
            b = torch.zeros(1,4)
            i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
            b.zero_()[:,i] = 1
            b=b.view(1,2,2)
        else:
            b = 1-b
        maskList.append(b)
    maskList = torch.cat(maskList,0).to(torch.float32)
    BigList.append(maskList)

#mask = torch.tensor([[[1,0],[1,0]]])
#maskList = torch.cat([mask if no%2==0 else 1-mask  for no in range(4)], 0).to(dtype=torch.float32)

#slayer = [utils.SimpleMLPreshape([4,32,4],[nn.ReLU(),utils.ScalableTanh(4)]) for _ in range(4)]
#tlayer = [utils.SimpleMLPreshape([4,32,4]) for _ in range(4)]
layers = [flow.RNVP(BigList[n], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),None]) for _ in range(4)], [utils.SimpleMLPreshape([4,32,32,4],[nn.ELU(),nn.ELU(),utils.ScalableTanh(4)]) for _ in range(4)]
) for n in range(2*2*2)]

length = 4
repeat = 2

t = flow.MERA(2,length,layers,repeat,p)

#t = flow.RNVP(maskList, tlayer,slayer,p)

def op(x):
    return -x

sym = [op]

m = train.Symmetrized(t, sym)
#m = Symmetrize(t)

target = source.Ising(4,2,2.269185314213022)
epochs = 1000
batchSize =32
loss,_,_ = traint()
loss = torch.Tensor(loss).reshape(-1)
print(loss[-10:].mean())

def latentU(z):
    x,_ = m.generate(z)
    return -(m.prior.logProbability(z)+target.logProbability(x)-m.logProbability(x))

batchSize = 256
z_ = m.prior.sample(batchSize)

z_,zaccept = utils.HMCwithAccept(latentU,z_,100,5,0.1)
x_z,_ = m.generate(z_)
def measure(x):
    p = torch.sigmoid(2.*x).view(-1, target.nvars[0])
    #sample spin
    #s = 2*torch.bernoulli(p).data.numpy()-1
    #sf = (s.mean(axis=1))**2
    #for i in range(s.shape[0]):
    #    print (' '.join(map(str, s[i,:])))
    
    #improved estimator
    s = 2.*p.data.cpu().numpy() - 1. 
    #en = -(np.dot(s, target.K) * s).mean(axis= 1) # energy
    sf = (s.mean(axis=1))**2 - (s**2).sum(axis=1)/target.nvars[0]**2  +1./target.nvars[0] #structure factor
    return  sf
obs = measure(x_z)
print("obs_z:",obs.mean(),  ' +/- ' , obs.std()/np.sqrt(1.*batchSize))

import pdb
pdb.set_trace()
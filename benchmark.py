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
    return train.learn(target,m,batchSize,epochs)

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

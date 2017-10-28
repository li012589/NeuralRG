import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 

from model.realnvp import RealNVP 
from train.objectives import ring2d

def _accept(e1,e2):
   diff = e1-e2
   return diff.exp()-diff.uniform_()>=0.0

class MCMC:
    def __init__(self, nvars, batchsize, logp, model=None):
        
        self.batchsize = batchsize
        self.nvars = nvars
        self.logp = logp 
        self.model = model

        self.x = torch.randn(self.batchsize, self.nvars)
        
    def run(self,ntherm,nmeasure,nskip):
        self.nmeasure= nmeasure 
        self.ntherm = ntherm
            
        for n in range(ntherm): 
            self.step()
        
        for n in range(nmeasure):
            for i in range(nskip):
                self.step()
            self.measure()

    def step(self):
        #x = torch.randn(self.batchsize, self.nvars)
        #x = self.model.backward(z)
        #ratio = torch.exp(self.logp(x) - self.logp(self.x))
        #print(ratio)

        #print(x)
        #print(self.x.reshape(1,-1))
        #print(torch.from_numpy(self.x.reshape(1, -1)))
        #print(model.logp(Variable(torch.from_numpy(self.x.reshape(1, -1)))).data.numpy()[0])
        #print(model.logp(x).data.numpy()[0])
        #print(self.logp(x.data.numpy().reshape(2,)))
        #print(self.logp(self.x))

        #ratio = np.exp(  model.logp(Variable(torch.from_numpy(self.x.reshape(1, -1)))).data.numpy()[0]
        #               - model.logp(x).data.numpy()[0]
        #               + self.logp(x.data.numpy().reshape(2,)) 
        #               - self.logp(self.x)
        #              )

        x = torch.randn(self.batchsize, self.nvars)
        accept = _accept(self.logp(x), self.logp(self.x)).view(self.batchsize, -1)
        accept = torch.cat((accept, accept), 1)
        reject = 1-accept 
        self.x = self.x * reject.float() + x* accept.float()

    def measure(self):
        x = self.x.numpy()
        logp = self.logp(self.x)
        for i in range(x.shape[0]):
            print (x[i, 0], x[i, 1], logp[i])

if __name__ == '__main__': 
    from numpy.random.mtrand import RandomState 

    #import argparse
    #parser = argparse.ArgumentParser(description='')
    #group = parser.add_argument_group('group')
    #group.add_argument("-Nvars", type=int,  help="")
    #group.add_argument("-Nlayers", type=int,  help="")
    #group.add_argument("-Hs", type=int,  help="")
    #group.add_argument("-Ht", type=int,  help="")
    #args = parser.parse_args()
    
    #construct model 
    #model = RealNVP(Nvars = args.Nvars, 
    #                Nlayers = args.Nlayers, 
    #                Hs = args.Hs, 
    #                Ht = args.Ht)
    #model.load_state_dict(torch.load(model.name))
    
    nvars = 2
    batchsize = 100
    mcmc = MCMC(nvars, batchsize, ring2d) #, model=model)
    mcmc.run(0, 100, 10)


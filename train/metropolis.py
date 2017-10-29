import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 

from model.realnvp import RealNVP 
from train.objectives import ring2d

__all__ = ["MCMC"]

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
        
        accratio = 0.0 
        for n in range(nmeasure):
            for i in range(nskip):
                accratio += self.step()
            self.measure()
        print ('#accratio:', accratio/float(nmeasure*nskip)) 

    def step(self):
        #TODO: change this if to something more elegant  
        if self.model is None:
            #no model 
            x = torch.randn(self.batchsize, self.nvars)
            accept = _accept(self.logp(x)+(x**2).sum(dim=1)/2., 
                             self.logp(self.x)+(self.x**2).sum(dim=1)/2)
        else: 
            #use a model 
            z = Variable(torch.randn(self.batchsize, self.nvars), volatile=True) # prior 
            x = self.model.backward(z)

            accept = _accept(self.logp(x.data)-self.model.logp(x).data, 
                             self.logp(self.x)-self.model.logp(Variable(self.x, volatile=True)).data)

        accratio = accept.float().mean()
        accept = accept.view(self.batchsize, -1)
        accept = torch.cat((accept, accept), 1)
        reject = 1-accept 
        
        #TODO: try to avoid this if 
        if self.model is None:
            self.x = self.x * reject.float() + x* accept.float()
        else:
            self.x = self.x * reject.float() + x.data* accept.float()

        return accratio

    def measure(self):
        x = self.x.numpy()
        logp = self.logp(self.x)
        for i in range(x.shape[0]):
            print (x[i, 0], x[i, 1], logp[i])

if __name__ == '__main__': 
    from numpy.random.mtrand import RandomState 
    
    #TODO: use a model or not in a more elegant way 
    import argparse
    parser = argparse.ArgumentParser(description='')
    group = parser.add_argument_group('group')
    group.add_argument("-Nvars", type=int,  help="")
    group.add_argument("-Nlayers", type=int,  help="")
    group.add_argument("-Hs", type=int,  help="")
    group.add_argument("-Ht", type=int,  help="")
    args = parser.parse_args()
    
    #construct model 
    model = RealNVP(Nvars = args.Nvars, 
                    Nlayers = args.Nlayers, 
                    Hs = args.Hs, 
                    Ht = args.Ht)
    model.load_state_dict(torch.load(model.name))
    
    nvars = 2
    batchsize = 100
    mcmc = MCMC(nvars, batchsize, ring2d, model=None)
    mcmc.run(0, 100, 100)


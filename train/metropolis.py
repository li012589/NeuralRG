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
    """
    Markov Chain Monte Carlo 

    Args:
        nvars (int): number of variables.
        batchsize (int): batch size.
        logp (function): target log-probability.
        model (): model used for update proposal.
    """

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
        accept = torch.cat((accept, accept), 1) # well, this assumes nvars = 2 
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
    import os, sys 
    import argparse

    parser = argparse.ArgumentParser(description='')
    group = parser.add_argument_group('group')
    group.add_argument("-Nvars", type=int, default=2, help="")
    group.add_argument("-Batchsize", type=int, default=100, help="")
    group.add_argument("-loadmodel", action='store_true', help="load model")

    group.add_argument("-Nlayers", type=int, default=4, help="")
    group.add_argument("-Hs", type=int, default=10, help="")
    group.add_argument("-Ht", type=int, default=10, help="")
    args = parser.parse_args()

   
    if args.loadmodel:
        #construct model 
        model = RealNVP(Nvars = args.Nvars, 
                        Nlayers = args.Nlayers, 
                        Hs = args.Hs, 
                        Ht = args.Ht)
        try: 
            model.load_state_dict(torch.load(model.name))
        except FileNotFoundError:
            print ('model file not found:', model.name)
            sys.exit(1) # exit, otherwise we will continue newly constructed real NVP model 
    else:
        #start from a fresh model 
        model = None
    
    mcmc = MCMC(args.Nvars, args.Batchsize, ring2d, model=model)
    mcmc.run(0, 100, 10)

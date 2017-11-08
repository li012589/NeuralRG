if __name__ =="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 

from model import Gaussian,MLP,RealNVP
from train.objectives import Ring2D, Ring5, Wave 

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
        target (Target): target log-probability.
        model (): model used for update proposal.
    """

    def __init__(self, nvars, batchsize, target, model, usemodel):
        
        self.batchsize = batchsize
        self.nvars = nvars
        self.target = target 
        self.model = model
        self.usemodel = usemodel

        self.x = torch.randn(self.batchsize, self.nvars)
        self.measurements = []
        
    def run(self,ntherm,nmeasure,nskip):
        self.nmeasure= nmeasure 
        self.ntherm = ntherm
            
        for n in range(ntherm): 
            self.step()
        
        self.accratio = 0.0 
        for n in range(nmeasure):
            for i in range(nskip):
                self.accratio += self.step()
            self.measure()
        self.accratio /= float(nmeasure*nskip) 

        print ('#accratio:', self.accratio)

    def step(self):
            
        #sample prior 
        z = model.prior(self.batchsize, volatile=True)

        if self.usemodel: 
            x = self.model.generate(z) # use the model to generate sample

            accept = _accept(self.target(x.data)-self.model.logProbability(x).data, 
                             self.target(self.x)-self.model.logProbability(Variable(self.x, volatile=True)).data)

        else:
            x = z                      # pass prior directly to the output
            accept = _accept(self.target(x.data)-self.model.prior.logProbability(x).data, 
                             self.target(self.x)-self.model.prior.logProbability(Variable(self.x, volatile=True)).data)


        accratio = accept.float().mean()
        accept = accept.view(self.batchsize, -1)
        
        self.x.masked_scatter_(accept, torch.masked_select(x.data, accept))

        return accratio

    def measure(self):

        x = self.x.numpy()
        logp = self.target(self.x)
        for i in range(x.shape[0]):
            print (x[i, 0], x[i, 1], logp[i])
        self.measurements.append( self.target.measure(self.x) )

if __name__ == '__main__': 
    import os, sys 
    import h5py 
    import argparse
    import subprocess 

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nvars", type=int, default=2, help="")
    parser.add_argument("-Batchsize", type=int, default=100, help="")
    parser.add_argument("-loadmodel", action='store_true', help="load model")
    parser.add_argument("-modelname", default=None, help="model name")
    parser.add_argument("-target", default='ring2d', help="target distribution")
    parser.add_argument("-folder", default='data/', help="where to store results")

    group = parser.add_argument_group('network parameters')
    group.add_argument("-Nlayers", type=int, default=8, help="")
    group.add_argument("-Hs", type=int, default=10, help="")
    group.add_argument("-Ht", type=int, default=10, help="")
    args = parser.parse_args()

    gaussian = Gaussian([args.Nvars])

    sList = [MLP(args.Nvars//2, args.Hs) for _ in range(args.Nlayers)]
    tList = [MLP(args.Nvars//2, args.Ht) for _ in range(args.Nlayers)] 

    #start from a fresh model 
    #if args.loadmodel = False, we actually only use its prior 
    model = RealNVP([args.Nvars], sList, tList, gaussian, name=args.modelname)
   
    if args.loadmodel:
        try:
            saveDict = torch.load('./'+model.name)
            model.loadModel(saveDict)
            print ('load model', model.name)
        except FileNotFoundError:
            print ('model file not found:', model.name)
            sys.exit(1) # exit, otherwise we will continue newly constructed real NVP model 
    
    if args.target == 'ring2d':
        target = Ring2D()
    elif args.target == 'ring5':
        target = Ring5()
    elif args.target == 'wave':
        target = Wave()
    else:
        print ('what target ?', args.target)
        sys.exit(1) 
  
    mcmc = MCMC(args.Nvars, args.Batchsize, target, model, usemodel=args.loadmodel)
    mcmc.run(0, 1000, 1)

    # store results
    # TODO: use replace later 
    cmd = ['mkdir', '-p', args.folder]
    subprocess.check_call(cmd)
    key = args.target \
         +'_Nl' + str(args.Nlayers) \
         +'_Hs' + str(args.Hs) \
         +'_Ht' + str(args.Ht) 
    key += '_mc'

    h5 = h5py.File(args.folder +'/'+key+'.h5','w')
    params = h5.create_group('params')
    params.create_dataset("Nvars", data=args.Nvars)
    params.create_dataset("Nlayers", data=args.Nlayers)
    params.create_dataset("Hs", data=args.Hs)
    params.create_dataset("Ht", data=args.Nlayers)
    params.create_dataset("target", data=args.target)
    params.create_dataset("loadmodel", data=args.loadmodel)
    params.create_dataset("model", data=model.name)

    results = h5.create_group('results')
    results.create_dataset("obs",data=np.array(mcmc.measurements))
    results.create_dataset("accratio",data=mcmc.accratio)
    h5.close()

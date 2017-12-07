import torch
torch.manual_seed(42)
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np

from model import Gaussian, MLP, RealNVP
from train.objectives import Ring2D, Ring5, Wave, Ising

__all__ = ["MCMC"]


class MCMC:
    """
    Markov Chain Monte Carlo

    Args:
        nvars (int): number of variables.
        batchsize (int): batch size.
        target (Target): target log-probability.
        prior (nn.Module): sampler.
        collectdata (bool): if to collect all data generated.
    """

    def __init__(self, target, prior, collectdata=False):
        """
        Init MCMC class
        Args:
            batchSize (int): batch size.
            target (Target): target log-probability.
            prior (sampler): sampler.
            collectdata (bool): if to collect all data generated.
        """
        self.target = target
        self.prior = prior
        self.collectdata = collectdata

        self.measurements = []

        if self.collectdata:
            self.data = []
    
    def run(self, batchSize,ntherm, nmeasure, nskip):
        """
        This method start sampling.
        Args:
            ntherm (int): number of steps used in thermalize.
            nmeasure (int): number of steps used in measure.
            nskip (int): number of steps skiped in measure.
        """
        z = self.prior.sample(batchSize)
        for n in range(ntherm):
            _,_,z = self.step(batchSize,z)

        zpack = []
        measurePack = []
        accratio = 0.0
        res = Variable(torch.DoubleTensor(batchSize).zero_())
        for n in range(nmeasure):
            for i in range(nskip):
                a,r,z = self.step(batchSize,z)
                accratio += a
                res += r
            if self.collectdata:
                z_ = z.data.cpu().numpy()
                #for i in range(z_.shape[0]):
                #    print (' '.join(map(str, z_[i,:])))
                logp = self.target(z).data.cpu().numpy()
                logp.shape = (-1, 1)
                zpack.append(np.concatenate((z_, logp), axis=1))
            measure = self.measure(z)
            measurePack.append(measure)
        accratio /= float(nmeasure * nskip)
        res /= float(nmeasure * nskip)

        #print ('#accratio:', accratio)
        return zpack,measurePack,accratio,res 

    def step(self,batchSize,z):
        """
        This method run a step of sampling.
        """
        x = self.prior.sample(batchSize)
        
        #print (type(x), type(z))
        #print ('pix', type(self.target(x)))
        #print ('px', type(self.prior.logProbability(x))) 
        #print ('piz', type(self.target(z)))
        #print ('pz', type(self.prior.logProbability(z)))

        #print ('pix', self.target(x).data)
        #print ('px', self.prior.logProbability(x).data) 
        #print ('piz', self.target(z).data)
        #print ('pz', self.prior.logProbability(z).data)

        pi_x = self.target(x)
        p_x = self.prior.logProbability(x)
        pi_z = self.target(z)
        p_z = self.prior.logProbability(z)

        diff = pi_x-pi_z -p_x + p_z
        accept = (diff.exp() >= Variable(torch.rand(diff.data.shape[0]).double()))

        a = accept.data.double().mean()
        r = -F.relu(-diff)

        #print ('#', accept.float().mean())
        #print (A.mean())

        accept = accept.view(batchSize, -1)
        #x.masked_scatter_(accept,torch.masked_select(z,accept))
        accept.data = accept.data.double()
        x = accept * x + (1.-accept)*z  # this is not inplace 
        return a,r,x

    def measure(self, x,measureFn=None):
        """
        This method measures some varibales.
        Args:
            measureFn (function): function to measure variables. If None, will run measure at target class.
        """
        if measureFn is None:
            measurements = self.target.measure(x)
        else:
            measurements = measureFn(x)
        return measurements


if __name__ == '__main__':
    pass

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
    
    @staticmethod
    def _reject(e1, e2):
        """
        This method gives if or not update x.
        Args:
            e1 (torch.Tensor): energy of original x.
            e2 (torch.Tensor): energy of proposed x.
        Return:
            ifUpdate (bool): if update x.
        """
        diff = e1 - e2
        return diff.exp() - Variable(torch.rand(diff.data.shape[0]).double()) < 0.0

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

        reject = self._reject(
                        (self.target(x)) - (self.prior.logProbability(x)),
                        (self.target(z)) - (self.prior.logProbability(z))
                    )

        a = 1.-reject.data.double().mean()

        r =-F.relu( - self.target(x)
                    + self.prior.logProbability(x)  
                    + self.target(z)
                    - self.prior.logProbability(z) 
            )

        #print ('#', accept.float().mean())
        #print (A.mean())

        reject = reject.view(batchSize, -1)
        x.data.masked_scatter_(reject.data, torch.masked_select(z.data, reject.data))
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

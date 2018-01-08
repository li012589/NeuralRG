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
        model (nn.Module): sampler.
        collectdata (bool): if to collect all data generated.
    """

    def __init__(self, target, model, collectdata=False):
        """
        Init MCMC class
        Args:
            batchSize (int): batch size.
            target (Target): target log-probability.
            model(sampler): sampler.
            collectdata (bool): if to collect all data generated.
        """
        self.target = target
        self.model= model 
        self.collectdata = collectdata

        self.measurements = []

        if self.collectdata:
            self.data = []
   
    def run(self, batchSize,ntherm, nmeasure, nskip, cuda = None, double = False):
        """
        This method start sampling.
        Args:
            ntherm (int): number of steps used in thermalize.
            nmeasure (int): number of steps used in measure.
            nskip (int): number of steps skiped in measure.
            z (): initial state 
        """

        z = self.model.sample(batchSize)      # sample from model

        if double:
            kld = Variable(torch.DoubleTensor(batchSize).zero_())
            res = Variable(torch.DoubleTensor(batchSize).zero_())
        else:
            kld = Variable(torch.FloatTensor(batchSize).zero_())
            res = Variable(torch.FloatTensor(batchSize).zero_())
        if cuda is not None:
            z = z.cuda(cuda)
            kld = kld.cuda(cuda)
            res = res.cuda(cuda)
        zpack = [] # samples 
        xpack = [] # proposals
        measurepack = []
        accratio = 0.0
        for n in range(ntherm+nmeasure):
            for _ in range(nskip):
                _,_,_,z = self.step(batchSize,z,double)

            a,r,x,z = self.step(batchSize,z,double)

            if n>=ntherm:
                accratio += a # mean acceptance ratio 
                res += r      # log(A)
                #print ('sjd', squared_jumped_distance)
                #sjd += squared_jumped_distance 
                kld += self.model.logProbability(x)-self.target(x) # KL(p||\pi)

                measurepack.append(self.measure(z))
                if self.collectdata:
                    #collect samples
                    z_ = z.data.view(batchSize, -1)
                    #for i in range(z_.shape[0]):
                    #    print (' '.join(map(str, z_[i,:])))
                    logp = self.target(z).data
                    logp = logp.view(-1,1)
                    zpack.append(torch.cat((z_, logp), 1))
                    
                    #collect proposals 
                    x_ = x.data.view(batchSize, -1)
                    logp = self.target(x).data
                    logp = logp.view(-1, 1)
                    xpack.append(torch.cat((x_, logp), 1))
            

        accratio /= float(nmeasure)
        res /= float(nmeasure)
        kld /= float(nmeasure)
        zpack = torch.stack(zpack,0)
        xpack = torch.stack(xpack,0)

        #print ('#accratio:', accratio)
        return zpack,xpack,measurepack,accratio,res,kld

    def step(self,batchSize,z,double):
        """
        This method run a step of sampling.
        """
        x = self.model.sample(batchSize)
        
        #print (type(x), type(z))
        #print ('pix', type(self.target(x)))
        #print ('px', type(self.model.logProbability(x))) 
        #print ('piz', type(self.target(z)))
        #print ('pz', type(self.model.logProbability(z)))

        #print ('pix', self.target(x).data)
        #print ('px', self.model.logProbability(x).data) 
        #print ('piz', self.target(z).data)
        #print ('pz', self.model.logProbability(z).data)

        pi_x = self.target(x) # API change: should be self.target.logProbability()
        p_x = self.model.logProbability(x)
        pi_z = self.target(z)
        p_z = self.model.logProbability(z)
        #print(p_z.data.shape)
        #print(p_x.data.shape)
        #print(pi_x.data.shape)
        #print(pi_z.data.shape)
        diff = (pi_x-pi_z -p_x + p_z)
        #print ('diff', diff)
        r = -F.relu(-diff)
        accept = Variable(diff.data.exp() >= diff.data.uniform_())

        if double:
            a = accept.data.double().mean()
            accept.data = accept.data.double()
        else:
            a = accept.data.float().mean()
            accept.data = accept.data.float()
        accept = accept.view(batchSize, -1)

        z = accept * x.view(batchSize, -1) + (1.-accept)*z.view(batchSize, -1)
        z = z.view(x.data.shape)

        return a,r,x,z

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

import torch
torch.manual_seed(42)
from torch.autograd import Variable
import numpy as np

from model import Gaussian, MLP, RealNVP
from train.objectives import Ring2D, Ring5, Wave

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

    def __init__(self, target, prior, collectdata):
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
    def _accept(e1, e2):
        """
        This method gives if or not update x.
        Args:
            e1 (torch.Tensor): energy of original x.
            e2 (torch.Tensor): energy of proposed x.
        Return:
            ifUpdate (bool): if update x.
        """
        diff = e1 - e2
        return diff.exp() - diff.uniform_() >= 0.0

    def run(self, batchSize,ntherm, nmeasure, nskip):
        """
        This method start sampling.
        Args:
            ntherm (int): number of steps used in thermalize.
            nmeasure (int): number of steps used in measure.
            nskip (int): number of steps skiped in measure.
        """
        z = self.prior.sample(batchSize).data
        for n in range(ntherm):
            _,z = self.step(batchSize,z)

        zpack = []
        measurePack = []
        accratio = 0.0
        for n in range(nmeasure):
            for i in range(nskip):
                a,z = self.step(batchSize,z)
                #zpack.append(z)
                accratio += a
            if self.collectdata:
                z_ = z.cpu().numpy()
                logp = self.target(z).cpu().numpy()
                logp.shape = (-1, 1)
                zpack.append(np.concatenate((z_, logp), axis=1))
            measure = self.measure(z)
            measurePack.append(measure)
        accratio /= float(nmeasure * nskip)

        print ('#accratio:', accratio)
        return zpack,measurePack,accratio

    def step(self,batchSize,z):
        """
        This method run a step of sampling.
        """
        x = self.prior.sample(batchSize)
        accept = self._accept(
            self.target(x.data) - self.prior.logProbability(x).data,
            self.target(z) - self.prior.logProbability(Variable(z, volatile=True)).data)

        accratio = accept.float().mean()
        accept = 1-accept.view(batchSize, -1)

        x.data.masked_scatter_(accept, torch.masked_select(z, accept))

        return accratio,x.data

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

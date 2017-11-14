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

    def __init__(self, batchsize, target, prior, collectdata):
        """
        Init MCMC class
        Args:
            batchSize (int): batch size.
            target (Target): target log-probability.
            prior (sampler): sampler.
            collectdata (bool): if to collect all data generated.
        """
        self.batchsize = batchsize
        self.target = target
        self.prior = prior
        self.collectdata = collectdata
        self.x = self.prior(self.batchsize).data

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

    def run(self, ntherm, nmeasure, nskip):
        """
        This method start sampling.
        Args:
            ntherm (int): number of steps used in thermalize.
            nmeasure (int): number of steps used in measure.
            nskip (int): number of steps skiped in measure.
        """
        self.nmeasure = nmeasure
        self.ntherm = ntherm

        for n in range(ntherm):
            self.step()

        self.accratio = 0.0
        for n in range(nmeasure):
            for i in range(nskip):
                self.accratio += self.step()
            self.measure()
        self.accratio /= float(nmeasure * nskip)

        print ('#accratio:', self.accratio)

    def step(self):
        """
        This method run a step of sampling.
        """
        x = self.prior(self.batchsize)
        accept = self._accept(
            self.target(x.data) - self.prior.logProbability(x).data,
            self.target(self.x) - self.prior.logProbability(Variable(self.x, volatile=True)).data)

        accratio = accept.float().mean()
        accept = accept.view(self.batchsize, -1)

        self.x.masked_scatter_(accept, torch.masked_select(x.data, accept))

        return accratio

    def measure(self, measureFn=None):
        """
        This method measures some varibales.
        Args:
            measureFn (function): function to measure variables. If None, will run measure at target class.
        """
        if self.collectdata:
            x = self.x.numpy()
            logp = self.target(self.x).numpy()
            logp.shape = (-1, 1)
            self.data.append(np.concatenate((x, logp), axis=1))

        if measureFn is None:
            self.measurements.append(self.target.measure(self.x))
        else:
            self.measurements.append(measureFn(self.x))


if __name__ == '__main__':
    pass

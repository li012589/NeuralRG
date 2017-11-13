if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
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

        # print ('#accratio:', self.accratio)

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
    import os
    import sys
    import h5py
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-target", default='ring2d',
                        help="target distribution")
    parser.add_argument("-collectdata", action='store_true',
                        help="collect data")
    parser.add_argument("-folder", default='data/',
                        help="where to store results")
    parser.add_argument("-savename", default=None, help="")

    group = parser.add_argument_group('mc parameters')
    group.add_argument("-Batchsize", type=int, default=16, help="")
    group.add_argument("-Nsamples", type=int, default=1000, help="")
    group.add_argument("-Nskips", type=int, default=1, help="")

    group = parser.add_argument_group('network parameters')
    group.add_argument("-Loadname", default=None, help="")
    group.add_argument("-modelname", default=None, help="")
    group.add_argument("-Nlayers", type=int, default=8, help="")
    group.add_argument("-Hs", type=int, default=10, help="")
    group.add_argument("-Ht", type=int, default=10, help="")
    args = parser.parse_args()

    if args.target == 'ring2d':
        target = Ring2D()
    elif args.target == 'ring5':
        target = Ring5()
    elif args.target == 'wave':
        target = Wave()
    elif args.target == 'phi4':
        target = Phi4(3, 2, 1.0, 1.0)
    else:
        print('what target ?', args.target)
        sys.exit(1)

    gaussian = Gaussian([target.nvars])

    if args.Loadname is None:
        model = gaussian
        print("using gaussian")
    else:
        sList = [MLP(target.nvars // 2, args.Hs) for _ in range(args.Nlayers)]
        tList = [MLP(target.nvars // 2, args.Ht) for _ in range(args.Nlayers)]

        model = RealNVP([target.nvars], sList, tList, gaussian, name=None)
        try:
            model.loadModel(torch.load(args.Loadname))
            print('#load model', args.Loadname)
        except FileNotFoundError:
            print('model file not found:', args.Loadname)
        print("using model")
    mcmc = MCMC(args.Batchsize, target, model, collectdata=args.collectdata)
    mcmc.run(0, args.Nsamples, args.Nskips)
    cmd = ['mkdir', '-p', args.folder]
    subprocess.check_call(cmd)
    if args.savename is None:
        key = args.folder \
            + args.target \
            + '_Nl' + str(args.Nlayers) \
            + '_Hs' + str(args.Hs) \
            + '_Ht' + str(args.Ht)
    else:
        key = args.savename
    h5filename = key + '_mc.h5'
    print("save at: " + h5filename)
    h5 = h5py.File(h5filename, 'w')
    params = h5.create_group('params')
    params.create_dataset("Nvars", data=target.nvars)
    params.create_dataset("Nlayers", data=args.Nlayers)
    params.create_dataset("Hs", data=args.Hs)
    params.create_dataset("Ht", data=args.Ht)
    params.create_dataset("target", data=args.target)
    params.create_dataset("model", data=model.name)
    results = h5.create_group('results')
    results.create_dataset("obs", data=np.array(mcmc.measurements))
    results.create_dataset("accratio", data=mcmc.accratio)
    if args.collectdata:
        results.create_dataset("samples", data=mcmc.data)
    h5.close()

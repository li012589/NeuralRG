import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class PriorTemplate(torch.nn.Module):
    """

    This is the template class for prior, which will be used in realNVP class.
    Args:
        name (PriorTemplate): name of this prior.

    """

    def __init__(self, shapeList, double = False, name="prior"):
        super(PriorTemplate, self).__init__()

        """

        This method initialise this class.
        Args:
            name (PriorTemplate): name of this prior.

        """
        self.shapeList = shapeList
        self.name = name
        self.cudaNo = None
        self.double = double

    def __call__(self):
        """

        This method should return sampled variables in prior distribution.

        """
        raise NotImplementedError(str(type(self)))

    def logProbability(self, x):
        """

        This method should return the probability of input variable in prior distribution.

        """
        raise NotImplementedError(str(type(self)))

class GMM(PriorTemplate):
    """

    Gaussian mixture prior distribution.
    Args:
        name (PriorTemplate): name of this prior.
        shapeList (int list): shape of sampled variables.

    """

    def __init__(self, shapeList, double = False, name="GMM"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(GMM, self).__init__(shapeList, double ,name)
        assert len(shapeList) == 1
        self.mu1= torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.logsigma1 = torch.nn.Parameter(torch.FloatTensor([0.]), requires_grad=True)
        self.mu2 = torch.nn.Parameter(torch.FloatTensor([-1.]), requires_grad=True)
        self.logsigma2 = torch.nn.Parameter(torch.FloatTensor([0.]), requires_grad=True)

    def sample(self, batchSize, volatile=False):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        if self.mu1.is_cuda:
            selector = Variable(torch.from_numpy(np.random.choice(2, size=(batchSize,))).cuda(self.mu1.get_device()).type(self.mu1.data.type()).view(batchSize,-1))
            return selector * (Variable(torch.FloatTensor(*size).normal_().cuda(self.mu1.get_device()).type(self.mu1.data.type()))*torch.exp(self.logsigma1) + self.mu1) \
                 + (1.-selector)* (Variable(torch.FloatTensor(*size).normal_().cuda(self.mu1.get_device()).type(self.mu1.data.type()))*torch.exp(self.logsigma2) + self.mu2)
        else:
            selector = Variable(torch.from_numpy(np.random.choice(2, size=(batchSize,))).type(self.mu1.data.type()).view(batchSize,-1))
            return selector * (Variable(torch.FloatTensor(*size).normal_().type(self.mu1.data.type()))*torch.exp(self.logsigma1) + self.mu1) \
                 + (1.-selector)* (Variable(torch.FloatTensor(*size).normal_().type(self.mu1.data.type()))*torch.exp(self.logsigma2) + self.mu2)

    def __call__(self,*args,**kwargs):
        return self.sample(*args,**kwargs)

    def _log_normal(self, x, mu, sigma):
        '''
        log normal probability distribution 
        '''
        return (-0.5*((x-mu)/sigma)**2- 0.5* torch.log(2.*np.pi*sigma**2)).sum(dim=1)

    def logProbability(self, x):
        """

        This method gives the log probability of z in prior distribution.
        Args:
            z (torch.autograd.Variable): variables to get log probability of.
        Return:
            logProbability (torch.autograd.Variable): log probability of input variables.

        """
        return torch.log( 0.5* torch.exp(self._log_normal(x, self.mu1, torch.exp(self.logsigma1)))
                        + 0.5* torch.exp(self._log_normal(x, self.mu2, torch.exp(self.logsigma2)))
                        )

class Gaussian(PriorTemplate):
    """

    This is a class for Gaussian prior distribution.
    Args:
        name (PriorTemplate): name of this prior.
        shapeList (int list): shape of sampled variables.

    """

    def __init__(self, shapeList, sigma=1, requires_grad=False, double = False, name="gaussian"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(Gaussian, self).__init__(shapeList, double,name)
        self.sigma = torch.nn.Parameter(torch.FloatTensor([sigma]), requires_grad=requires_grad)
    def sample(self, batchSize, volatile=False):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        if self.sigma.is_cuda:
            return Variable(torch.randn(size).cuda(self.sigma.get_device()).type(self.sigma.data.type()))
        else:
            return Variable(torch.randn(size).type(self.sigma.data.type()))

    def __call__(self,*args,**kwargs):
        return self.sample(*args,**kwargs)

    def logProbability(self, z):
        """

        This method gives the log probability of z in prior distribution.
        Args:
            z (torch.autograd.Variable): variables to get log probability of.
        Return:
            logProbability (torch.autograd.Variable): log probability of input variables.

        """
        tmp = -0.5 * (z/self.sigma)**2 -0.5* torch.log(2.*np.pi*self.sigma**2)
        return tmp.view(z.data.shape[0],-1).sum(dim=1)  # sum all but the batch dimension

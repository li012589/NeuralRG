import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model import RealNVPtemplate, PriorTemplate

class Cauchy(PriorTemplate):
    """

    This is a class for Cauchy prior distribution.
    Args:
        name (PriorTemplate): name of this prior.
        shapeList (int list): shape of sampled variables.

    """

    def __init__(self, shapeList, sigma=1, requires_grad=False, name="cauchy"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(Cauchy, self).__init__(name)
        self.shapeList = shapeList
        self.sigma = torch.nn.Parameter(torch.DoubleTensor([sigma]), requires_grad= requires_grad)    

    def sample(self, batchSize, volatile=False, ifCuda=False, double=True):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        if ifCuda:
            if double:
                return Variable(torch.DoubleTensor(*size).cauchy_(sigma=self.sigma).pin_memory(),volatile=volatile)
            else:
                return Variable(torch.FloatTensor(*size).cauchy_(sigma=self.sigma).pin_memory(),volatile=volatile)
        else:
            if double:
                return Variable(torch.DoubleTensor(*size).cauchy_(sigma=self.sigma), volatile=volatile)
            else:
                return Variable(torch.FloatTensor(*size).cauchy_(sigma=self.sigma), volatile=volatile)
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
        tmp = -torch.log(z**2+self.sigma**2)
        return tmp.view(z.data.shape[0],-1).sum(dim=1)  # sum all but the batch dimension

class GMM(PriorTemplate):
    """

    Gaussian mixture prior distribution.
    Args:
        name (PriorTemplate): name of this prior.
        shapeList (int list): shape of sampled variables.

    """

    def __init__(self, shapeList, name="GMM"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(GMM, self).__init__(name)
        self.shapeList = shapeList
        
        #now we can only have two centers 

        #shared mu and sigma 
        self.mu1= torch.nn.Parameter(torch.DoubleTensor([1.]), requires_grad=True)    
        self.logsigma1 = torch.nn.Parameter(torch.DoubleTensor([0.]), requires_grad=True)    
        self.mu2 = torch.nn.Parameter(torch.DoubleTensor([-1.]), requires_grad=True)    
        self.logsigma2 = torch.nn.Parameter(torch.DoubleTensor([0.]), requires_grad=True)    
        
        #independed for each component 
        #self.mu1= torch.nn.Parameter(torch.DoubleTensor(*shapeList).normal_(), requires_grad=True)
        #self.logsigma1 = torch.nn.Parameter(torch.DoubleTensor(*shapeList).zero_(), requires_grad=True)
        #self.mu2 = torch.nn.Parameter(torch.DoubleTensor(*shapeList).normal_(), requires_grad=True)
        #self.logsigma2 = torch.nn.Parameter(torch.DoubleTensor(*shapeList).zero_(), requires_grad=True)

    def sample(self, batchSize, volatile=False, ifCuda=False, double=True):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        if ifCuda:
            raise NotImplementedError(str(type(self)))
        else:
            if double:
                #select the gaussian center 
                selector = torch.from_numpy(np.random.choice(2, size=(batchSize,))).double()
                selector = Variable(selector.view(batchSize, -1))
                #print (selector)
                
                #sample from one of the Gaussian 
                return selector * (Variable(torch.DoubleTensor(*size).normal_())*torch.exp(self.logsigma1) + self.mu1) \
                 + (1.-selector)* (Variable(torch.DoubleTensor(*size).normal_())*torch.exp(self.logsigma2) + self.mu2) 
            else:
                raise NotImplementedError(str(type(self)))

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

    def __init__(self, shapeList, sigma=1, requires_grad=False, name="gaussian"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(Gaussian, self).__init__(name)
        self.shapeList = shapeList
        self.sigma = torch.nn.Parameter(torch.DoubleTensor([sigma]), requires_grad=requires_grad)    

    def sample(self, batchSize, volatile=False, ifCuda=False, double=True):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        sigma = self.sigma.cpu()
        if ifCuda:
            if double:
                return Variable(torch.randn(size).double().pin_memory(),volatile=volatile) * sigma
            else:
                return Variable(torch.randn(size).pin_memory(),volatile=volatile) * sigma.half()
        else:
            if double:
                return Variable(torch.randn(size).double(), volatile=volatile) * sigma 
            else:
                return Variable(torch.randn(size), volatile=volatile) * sigma.half()

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
        tmp = -0.5 * (z/self.sigma)**2 
        return tmp.view(z.data.shape[0],-1).sum(dim=1)  # sum all but the batch dimension

class RealNVP(RealNVPtemplate):
    """

    This is a  class for simple realNVP.
    Args:
        shapeList (int list): shape of variable coverted.
        sList (torch.nn.Module list): list of nerual networks in s funtion.
        tList (torch.nn.Module list): list of nerual networks in s funtion.
        prior (PriorTemplate): the prior distribution used.
        NumLayers (int): number of layers in sList and tList.
        _logjac (torch.autograd.Variable): log of jacobian, only avaible after _generate method are called.
        mask (torch.Tensor): mask to divide x into y0 and y1, only avaible when self.createMask is called.
        name (string): name of this class.

    """

    def __init__(self, shapeList, sList, tList, prior, masktypelist, name=None, double=True):
        """

        This mehtod initialise this class.
        Args:
            shapeList (int list): shape of variable coverted.
            sList (torch.nn.Module list): list of nerual networks in s funtion.
            tList (torch.nn.Module list): list of nerual networks in s funtion.
            prior (PriorTemplate): the prior distribution used.
            name (string): name of this class.

        """
        super(RealNVP, self).__init__(
            shapeList, sList, tList, prior, name,double)
        self.createMask(masktypelist)

    def createMask(self, masktypelist, ifByte=0, double = True):
        """

        This method create mask for x, and save it in self.mask for later use.
        Args:
            maskType (string): specify waht type of mask to create. "channel" or "checkerboard".
            ifByte (int): flag variable, tell if output variable should be ByteTensor or FloatTensor.
        Return:
            mask (torch.Tensor): mask to divide x into y0 and y1.

        """
        self.masks = []
        self.masks_ = []
        for masktype in masktypelist: 
            size = self.shapeList.copy()
            if masktype in ["updown", "leftright"]: 
                slicedim = 1 if masktype == "updown" else 2
                size[slicedim] = size[slicedim] // 2
                if double:
                    maskOne = torch.ones(size).double()
                    maskZero = torch.zeros(size).double()
                else:
                    maskOne = torch.ones(size)
                    maskZero = torch.zeros(size)
                mask = torch.cat([maskOne, maskZero], slicedim)
            
            elif masktype == "checkerboard":
                assert (size[1] % 2 == 0)
                assert (size[2] % 2 == 0)
                if double:
                    unit = torch.DoubleTensor([[1, 0], [0, 1]])
                else:
                    unit = torch.FloatTensor([[1, 0], [0, 1]])
                mask = (unit.repeat(
                        size[0], size[1] // 2, size[2] // 2))
            else:
                raise ValueError("maskType not known.")
            
            print ('mask:', mask)
            if ifByte:
                mask = mask.byte()
            if self.ifCuda:
                cudaNo = self.mask.get_device()
                mask = mask.pin_memory().cuda(cudaNo)

            self.masks.append(Variable(mask))
            self.masks_.append(Variable(1-mask))

            #self.register_buffer("mask",mask)
            #self.register_buffer("mask_",1-mask)
        #return mask

    def generate(self, z):
        """

        This method generate complex distribution using variables sampled from prior distribution.
        Args:
            z (torch.autograd.Variable): input Variable.
        Return:
            x (torch.autograd.Variable): output Variable.

        """
        return self._generate(z, self.masks, self.masks_)

    def inference(self, x):
        """

        This method inference prior distribution using variable sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
        Return:
            z (torch.autograd.Variable): output Variable.

        """
        return self._inference(x, self.masks, self.masks_)

    def logProbability(self, x):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
        Return:
            log-probability (torch.autograd.Variable): log-probability of x.

        """
        return self._logProbability(x, self.masks, self.masks_)

    def logProbabilityWithInference(self,x):
        z = self._inference(x, self.masks, self.masks_, True)
        return self.prior.logProbability(z) + self._inferenceLogjac,z

    def saveModel(self, saveDic):
        """

        This methods add contents to saveDic, which will be saved outside.
        Args:
            saveDic (dictionary): contents to save.
        Return:
            saveDic (dictionary): contents to save with nerual networks in this class.

        """
        self._saveModel(saveDic)
        #saveDic["masks"] = self.masks.cpu()  # Do check if exist !!
        #saveDic["masks_"] = self.masks_.cpu()
        saveDic["shapeList"] = self.shapeList
        return saveDic

    def loadModel(self, saveDic):
        """

        This method lookk for saved contents in saveDic and load them.
        Args:
            saveDic (dictionary): contents to load.
        Return:
            saveDic (dictionary): contents to load.

        """
        self._loadModel(saveDic)
        self.register_buffer("mask",saveDic["mask"])
        self.register_buffer("mask_",saveDic["mask_"])
        self.shapeList = saveDic["shapeList"]
        return saveDic

    def sample(self, batchSize,useGenerate=True):
        """

        This method directly sample samples of batch size given
        Args:
            batchSize (int): size of sampled batch.
        return:
            samples: (torch.autograd.Variable): output Variable.
        """
        if self.ifCuda:
            cudaNo = self.mask.get_device()
            z = self.prior(batchSize, ifCuda=True).cuda(cudaNo)
        else:
            z = self.prior(batchSize)
        if useGenerate:
            return self.generate(z)
        else:
            return self.inference(z)

if __name__ == "__main__":
    
    gmm = GMM([4])
    samples = gmm.sample(10)
    print (samples)
    print (gmm.logProbability(samples))


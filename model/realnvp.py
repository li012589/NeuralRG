import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .template import RealNVPtemplate

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

    def __init__(self, shapeList, sList, tList, prior, maskType="channel", double=False, mode = 0, name=None):
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
        if isinstance(maskType,str):
            maskType = [maskType] * self.NumLayers
        else:
            assert len(maskType) == self.NumLayers
        self.maskType = maskType
        self.createMask(maskType)
        self.mode = mode

    def createMask(self, maskType, ifByte=0, double = False):
        """

        This method create mask for x, and save it in self.mask for later use.
        Args:
            maskType (string): specify waht type of mask to create. "channel" or "checkerboard".
            ifByte (int): flag variable, tell if output variable should be ByteTensor or FloatTensor.
        Return:
            mask (torch.Tensor): mask to divide x into y0 and y1.

        """
        maskList = None
        mask_List = None
        for iterm in maskType:
            mask = self._createMaskMeta(iterm,ifByte,double)
            mask_ = 1 - mask
            if maskList is None:
                maskList = mask.view(1,*mask.shape)
                mask_List = mask_.view(1,*mask_.shape)
            else:
                maskList = torch.cat([maskList,mask.view(1,*mask.shape)],0)
                mask_List = torch.cat([mask_List,mask_.view(1,*mask_.shape)],0)
        self.register_buffer("mask",maskList)
        self.register_buffer("mask_",mask_List)

    def _createMaskMeta(self,maskType,ifByte, double):
        size = self.shapeList.copy()
        if maskType == "channel":
            size[0] = size[0] // 2
            if double:
                maskOne = torch.ones(size).double()
                maskZero = torch.zeros(size).double()
            else:
                maskOne = torch.ones(size)
                maskZero = torch.zeros(size)
            mask = torch.cat([maskOne, maskZero], 0)

        elif maskType == "checkerboard":
            assert (size[1] % 2 == 0)
            assert (size[2] % 2 == 0)
            if double:
                unit = torch.DoubleTensor([[1, 0], [0, 1]])
            else:
                unit = torch.FloatTensor([[1, 0], [0, 1]])
            mask = (unit.repeat(
                size[0], size[1] // 2, size[2] // 2))
        elif ('updown' in maskType) or ('leftright' in maskType):
            slicedim = 1 if ("updown" in maskType) else 2
            size[slicedim] = size[slicedim] // 2
            if double:
                maskOne = torch.ones(size).double()
                maskZero = torch.zeros(size).double()
            else:
                maskOne = torch.ones(size)
                maskZero = torch.zeros(size)
            mask = torch.cat([maskOne, maskZero], slicedim)
        elif ('bars' in maskType):
            assert (size[1] % 2 == 0)
            assert (size[2] % 2 == 0)
            if double:
                unit = torch.DoubleTensor([[1, 0], [1, 0]])
            else:
                unit = torch.FloatTensor([[1, 0], [1, 0]])
            mask = (unit.repeat(
                size[0], size[1] // 2, size[2] // 2))
        elif ('stripes' in maskType):
            assert (size[1] % 2 == 0)
            assert (size[2] % 2 == 0)
            if double:
                unit = torch.DoubleTensor([[1, 1], [0, 0]])
            else:
                unit = torch.FloatTensor([[1, 1], [0, 0]])
            mask = (unit.repeat(
                size[0], size[1] // 2, size[2] // 2))
        else:
            raise ValueError("maskType not known.")
        if ifByte:
            mask = mask.byte()
        if self.ifCuda:
            cudaNo = self.mask.get_device()
            mask = mask.pin_memory().cuda(cudaNo)
        #self.register_buffer("mask",mask)
        #self.register_buffer("mask_",1-mask)
        return mask

    def generate(self, *args, **kwargs):
        """

        This method generate complex distribution using variables sampled from prior distribution.
        Args:
            z (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            x (torch.autograd.Variable): output Variable.

        """
        if self.mode == 0 :
            return self._generate(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 1:
            return self._generateWithContraction(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 2:
            return self._generateWithSlice(*args, **kwargs)
        else:
            raise NotImplementedError("Unknown work mode for realnvp")

    def inference(self, *args, **kwargs):
        """

        This method inference prior distribution using variable sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            z (torch.autograd.Variable): output Variable.

        """
        if self.mode == 0 :
            return self._inference(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 1:
            return self._inferenceWithContraction(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 2:
            return self._inferenceWithSlice(*args, **kwargs)
        else:
            raise NotImplementedError("Unknown work mode for realnvp")

    def logProbability(self, *args, **kwargs):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            log-probability (torch.autograd.Variable): log-probability of x.

        """
        if self.mode == 0 :
            return self._logProbability(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 1:
            return self._logProbabilityWithContraction(*args, self.mask, self.mask_, **kwargs)
        elif self.mode == 2:
            return self._logProbabilityWithSlice(*args, **kwargs)
        else:
            raise NotImplementedError("Unknown work mode for realnvp")

    def logProbabilityWithInference(self,*args, **kwargs):
        kwargs['ifLogjac'] = True
        if self.mode == 0 :
            z = self._inference(*args, self.mask, self.mask_, **kwargs)
            return self.prior.logProbability(z) + self._inferenceLogjac,z
        elif self.mode == 1:
            z = self._inferenceWithContraction(*args, self.mask, self.mask_, **kwargs)
            return self.prior.logProbability(z) + self._inferenceLogjac,z
        elif self.mode == 2:
            z = self._inferenceWithSlice(*args, **kwargs)
            return self.prior.logProbability(z) + self._inferenceLogjac,z
        else:
            raise NotImplementedError("Unknown work mode for realnvp")

    def saveModel(self, saveDic):
        """

        This methods add contents to saveDic, which will be saved outside.
        Args:
            saveDic (dictionary): contents to save.
        Return:
            saveDic (dictionary): contents to save with nerual networks in this class.

        """
        self._saveModel(saveDic)
        saveDic["mask"] = self.mask.cpu()  # Do check if exist !!
        saveDic["mask_"] = self.mask_.cpu()
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

    def sample(self, batchSize, useGenerate=True):
        """

        This method directly sample samples of batch size given
        Args:
            batchSize (int): size of sampled batch.
            sliceDim (int): in which dimension should mask be used on y.
        return:
            samples: (torch.autograd.Variable): output Variable.
        """
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


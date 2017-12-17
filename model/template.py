import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import checkNan

class RealNVPtemplate(torch.nn.Module):
    """

    This is a template class for realNVP. This base class doesn't handle mask creating, saving and changing.
    Args:
        shapeList (int list): shape of variable coverted.
        sList (torch.nn.Module list): list of nerual networks in s funtion.
        tList (torch.nn.Module list): list of nerual networks in s funtion.
        prior (PriorTemplate): the prior distribution used.
        NumLayers (int): number of layers in sList and tList.
        _generateLogjac (torch.autograd.Variable): log of jacobian of generate function, only avaible after _generate method are called.
        _inferenceLogjac (torch.autograd.Variable): log of jacobian of inference function, only avaible after _inference method are called.
        name (string): name of this class.
        ifCuda (bool): if this instance will be on GPU or not.

    """

    def __init__(self, shapeList, sList, tList, prior, name=None, double=True):
        """

        This mehtod initialise this class.
        Args:
            shapeList (int list): shape of variable coverted.
            sList (torch.nn.Module list): list of nerual networks in s funtion.
            tList (torch.nn.Module list): list of nerual networks in s funtion.
            prior (PriorTemplate): the prior distribution used.
            name (string): name of this class.

        """
        super(RealNVPtemplate, self).__init__()

        assert len(tList) == len(tList)
        if double:
            self.tList = torch.nn.ModuleList(tList).double()
            self.sList = torch.nn.ModuleList(sList).double()
        else:
            self.tList = torch.nn.ModuleList(tList)
            self.sList = torch.nn.ModuleList(sList)
        self.NumLayers = len(self.tList)
        self.prior = prior
        self.shapeList = shapeList
        self.ifCuda = False
        self.pointer = "logProbability"
        if name is None:
            self.name = "realNVP_" + \
                str(self.NumLayers) + "inner_" + \
                "layers_" + self.prior.name + "Prior"
        else:
            self.name = name
        self.double = double

    def cuda(self,device=None,async=False):
        """

        This method move everything in RealNVPtemplate to GPU.
        Return:
            cudaModel (nn.Module.cuda): the instance in GPU.

        """
        cudaModel = super(RealNVPtemplate, self).cuda(device)
        cudaModel.ifCuda = True
        return cudaModel

    def cpu(self):
        """

        This method move everything in RealNVPtemplate to CPU.
        Return:
            cudaModel (nn.Module): the instance in CPU.

        """
        cpuModel = super(RealNVPtemplate, self).cpu()
        cpuModel.ifCuda = False
        return cpuModel

    def _generate(self, y, masks, masks_, ifLogjac=False):
        """

        This method generate complex distribution using variables sampled from prior distribution.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            y (torch.autograd.Variable): output Variable.

        """
        if self.ifCuda:
            cudaNo = y.get_device()
        if ifLogjac:
            if self.ifCuda:
                if self.double:
                    self._generateLogjac = Variable(
                        torch.zeros(y.data.shape[0]).double().pin_memory().cuda(cudaNo))
                else:
                    self._generateLogjac = Variable(
                        torch.zeros(y.data.shape[0]).pin_memory().cuda(cudaNo))
            else:
                if self.double:
                    self._generateLogjac = Variable(torch.zeros(y.data.shape[0]).double())
                else:
                    self._generateLogjac = Variable(torch.zeros(y.data.shape[0]))
        for i in range(self.NumLayers):
            if (i % 2 == 0):
                y_ = masks[i//2] * y
                s = self.sList[i](y_) * masks_[i//2]
                t = self.tList[i](y_) * masks_[i//2]
                #checkNan(s)
                y = y_ + masks_[i//2] * (y * checkNan(torch.exp(s)) + t)
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            else:
                y_ = masks_[i//2] * y
                s = self.sList[i](y_) * masks[i//2]
                t = self.tList[i](y_) * masks[i//2]
                #checkNan(s)
                y = y_ + masks[i//2] * (y * checkNan(torch.exp(s)) + t)
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
        return y

    def _inference(self, y, masks, masks_, ifLogjac=False):
        """

        This method inference prior distribution using variable sampled from complex distribution.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            y (torch.autograd.Variable): output Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.

        """
        if self.ifCuda:
            cudaNo = y.get_device()
        if ifLogjac:
            if self.ifCuda:
                if self.double:
                    self._inferenceLogjac = Variable(
                        torch.zeros(y.data.shape[0]).double().pin_memory().cuda(cudaNo))
                else:
                    self._inferenceLogjac = Variable(
                        torch.zeros(y.data.shape[0]).pin_memory().cuda(cudaNo))
            else:
                if self.double:
                    self._inferenceLogjac = Variable(torch.zeros(y.data.shape[0]).double())
                else:
                    self._inferenceLogjac = Variable(torch.zeros(y.data.shape[0]))

        for i in list(range(self.NumLayers))[::-1]:
            if (i % 2 == 0):
                y_ = masks[i//2] * y
                s = self.sList[i](y_) * masks_[i//2]
                t = self.tList[i](y_) * masks_[i//2]
                #checkNan(s)
                y = masks_[i//2] * (y - t) * checkNan(torch.exp(-s)) + y_
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            else:
                y_ = masks_[i//2] * y
                s = self.sList[i](y_) * masks[i//2]
                t = self.tList[i](y_) * masks[i//2]
                #checkNan(s)
                y = masks[i//2] * (y - t) * checkNan(torch.exp(-s)) + y_
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
        return y

    def _logProbability(self, x, masks, masks_):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
        Return:
            probability (torch.autograd.Variable): probability of x.

        """
        z = self._inference(x, masks, masks_, True)
        return self.prior.logProbability(z) + self._inferenceLogjac

    def _saveModel(self, saveDic):
        """

        This methods add contents to saveDic, which will be saved outside.
        Args:
            saveDic (dictionary): contents to save.
        Return:
            saveDic (dictionary): contents to save with nerual networks in this class.

        """
        # save is done some where else, adding s,t to the dict
        for i in range(self.NumLayers):
            saveDic["__" + str(i) + 'sLayer'] = self.sList[i].state_dict()
            saveDic["__" + str(i) + 'tLayer'] = self.tList[i].state_dict()
        return saveDic

    def _loadModel(self, saveDic):
        """

        This method lookk for saved contents in saveDic and load them.
        Args:
            saveDic (dictionary): contents to load.
        Return:
            saveDic (dictionary): contents to load.

        """
        # load is done some where else, pass the dict here.
        for i in range(self.NumLayers):
            self.sList[i].load_state_dict(saveDic["__" + str(i) + 'sLayer'])
            self.tList[i].load_state_dict(saveDic["__" + str(i) + 'tLayer'])
        return saveDic

    def forward(self,*args,**kwargs):
        return getattr(self,self.pointer)(*args,**kwargs)


class PriorTemplate(torch.nn.Module):
    """

    This is the template class for prior, which will be used in realNVP class.
    Args:
        name (PriorTemplate): name of this prior.

    """

    def __init__(self, name="prior"):
        super(PriorTemplate, self).__init__()

        """

        This method initialise this class.
        Args:
            name (PriorTemplate): name of this prior.

        """
        self.name = name

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


if __name__ == "__main__":

    pass

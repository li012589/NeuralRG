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

    def _generate(self, y, mask, mask_, ifLogjac=False):
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
        mask = Variable(mask)
        mask_ = Variable(mask_)
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
                y_ = mask[i] * y
                s = self.sList[i](y_) * mask_[i]
                t = self.tList[i](y_) * mask_[i]
                #checkNan(s)
                y = y_ + mask_[i] * (y * checkNan(torch.exp(s)) + t)
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            else:
                y_ = mask_[i] * y
                s = self.sList[i](y_) * mask[i]
                t = self.tList[i](y_) * mask[i]
                #checkNan(s)
                y = y_ + mask[i] * (y * checkNan(torch.exp(s)) + t)
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
        return y

    def _generateMeta(self, y0, y1, ifLogjac):
        """

        This method is used to compose generate method
        Args:
            y0 (torch.autograd.Variable): input Variable.
            y1 (torch.autograd.Variable): input Variable.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            y0 (torch.autograd.Variable): output Variable.
            y1 (torch.autograd.Variable): output Variable.

        """
        for i in range(self.NumLayers):
            if (i % 2 == 0):
                s = self.sList[i](y0)
                t = self.tList[i](y0)
                #checkNan(s)
                y1 = y1 * checkNan(torch.exp(s)) + t
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            else:
                s = self.sList[i](y1)
                t = self.tList[i](y1)
                #checkNan(s)
                y0 = y0 * checkNan(torch.exp(s)) + t
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
        return y0, y1

    def _generateWithContraction(self, y, mask, mask_, sliceDim, ifLogjac=False):
        """

        This method generate complex distribution using variables sampled from prior distribution. Unlike _generate method this method use mask first to contract y into y0 and y1 to reduce computational complexity.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            sliceDim (int): in which dimension should mask be used on y.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            output (torch.autograd.Variable): output Variable.

        """
        mask = Variable(mask)
        mask_ = Variable(mask_)
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
        size = [-1] + self.shapeList
        size[sliceDim + 1] = size[sliceDim + 1] // 2
        for i in range(self.NumLayers):
            y1 = torch.masked_select(y, mask_[i]).view(size)
            y0 = torch.masked_select(y, mask[i]).view(size)
            if (i % 2 == 0):
                s = self.sList[i](y0)
                t = self.tList[i](y0)
                #checkNan(s)
                y1 = y1 * checkNan(torch.exp(s)) + t
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            else:
                s = self.sList[i](y1)
                t = self.tList[i](y1)
                #checkNan(s)
                y0 = y0 * checkNan(torch.exp(s)) + t
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            if self.ifCuda:
                if self.double:
                    output = Variable(torch.zeros(y.data.shape).double().pin_memory().cuda(cudaNo))
                else:
                    output = Variable(torch.zeros(y.data.shape).pin_memory().cuda(cudaNo))
            else:
                if self.double:
                    output = Variable(torch.zeros(y.data.shape).double())
                else:
                    output = Variable(torch.zeros(y.data.shape))
            output.masked_scatter_(mask[i], y0)
            output.masked_scatter_(mask_[i], y1)
            y = output
        return y

    def _generateWithSlice(self, y, sliceDim, ifLogjac=False):
        """

        This method generate complex distribution using variables sampled from prior distribution. Unlike _generate method this method first slice y into y0 and y1 to reduce computational complexity.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            sliceDim (int): in which dimension should mask be used on y.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            output (torch.autograd.Variable): output Variable.

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
        y0 = y.narrow(sliceDim + 1, 0, self.shapeList[sliceDim] // 2)
        y1 = y.narrow(
            sliceDim + 1, self.shapeList[sliceDim] // 2, self.shapeList[sliceDim] - 1)
        y0, y1 = self._generateMeta(y0, y1, ifLogjac)
        return torch.cat((y0, y1), sliceDim + 1)

    def _inference(self, y, mask, mask_, ifLogjac=False):
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
        mask = Variable(mask)
        mask_ = Variable(mask_)
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
                y_ = mask * y
                s = self.sList[i](y_) * mask_
                t = self.tList[i](y_) * mask_
                #checkNan(s)
                y = mask_ * (y - t) * checkNan(torch.exp(-s)) + y_
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            else:
                y_ = mask_ * y
                s = self.sList[i](y_) * mask
                t = self.tList[i](y_) * mask
                #checkNan(s)
                y = mask * (y - t) * checkNan(torch.exp(-s)) + y_
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
        return y

    def _inferenceMeta(self, y0, y1, ifLogjac):
        """

        This method is used to compose inference method
        Args:
            y0 (torch.autograd.Variable): input Variable.
            y1 (torch.autograd.Variable): input Variable.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            y0 (torch.autograd.Variable): output Variable.
            y1 (torch.autograd.Variable): output Variable.

        """
        for i in list(range(self.NumLayers))[::-1]:
            if (i % 2 == 0):
                s = self.sList[i](y0)
                t = self.tList[i](y0)
                #checkNan(s)
                y1 = (y1 - t) * checkNan(torch.exp(-s))
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            else:
                s = self.sList[i](y1)
                t = self.tList[i](y1)
                #checkNan(s)
                y0 = (y0 - t) * checkNan(torch.exp(-s))
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
        return y0, y1

    def _inferenceWithContraction(self, y, mask, mask_, sliceDim, ifLogjac=False):
        """

        This method inference prior distribution using variables sampled from complex distribution. Unlike _inference method this method use mask first to contract y into y0 and y1 to reduce computational complexity.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            sliceDim (int): in which dimension should mask be used on y.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            output (torch.autograd.Variable): output Variable.

        """
        mask = Variable(mask)
        mask_ = Variable(mask_)
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
        size = [-1] + self.shapeList
        size[sliceDim + 1] = size[sliceDim + 1] // 2
        for i in list(range(self.NumLayers))[::-1]:
            y0 = torch.masked_select(y, mask[i]).view(size)
            y1 = torch.masked_select(y, mask_[i]).view(size)
            if (i % 2 == 0):
                s = self.sList[i](y0)
                t = self.tList[i](y0)
                #checkNan(s)
                y1 = (y1 - t) * checkNan(torch.exp(-s))
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            else:
                s = self.sList[i](y1)
                t = self.tList[i](y1)
                #checkNan(s)
                y0 = (y0 - t) * checkNan(torch.exp(-s))
                if ifLogjac:
                    for _ in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            if self.ifCuda:
                if self.double:
                    output = Variable(torch.zeros(y.data.shape).double().pin_memory().cuda(cudaNo))
                else:
                    output = Variable(torch.zeros(y.data.shape).pin_memory().cuda(cudaNo))
            else:
                if self.double:
                    output = Variable(torch.zeros(y.data.shape).double())
                else:
                    output = Variable(torch.zeros(y.data.shape))
                output.masked_scatter_(mask[i], y0)
                output.masked_scatter_(mask_[i], y1)
                y = output
        return y

    def _inferenceWithSlice(self, y, sliceDim, ifLogjac=False):
        """

        This method inference prior distribution using variables sampled from complex distribution. Unlike _inference method this method first slice y into y0 and y1 to reduce computational complexity.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            sliceDim (int): in which dimension should mask be used on y.
            ifLogjac (int): iflag variable, used to tell if log jacobian should be computed.
        Return:
            output (torch.autograd.Variable): output Variable.

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
        y0 = y.narrow(sliceDim + 1, 0, self.shapeList[sliceDim] // 2)
        y1 = y.narrow(
            sliceDim + 1, self.shapeList[sliceDim] // 2, self.shapeList[sliceDim] - 1)
        y0, y1 = self._inferenceMeta(y0, y1, ifLogjac)
        return torch.cat((y0, y1), sliceDim + 1)

    def _logProbability(self, x, mask, mask_):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
        Return:
            probability (torch.autograd.Variable): probability of x.

        """
        z = self._inference(x, mask, mask_, True)
        return self.prior.logProbability(z) + self._inferenceLogjac

    def _logProbabilityWithSlice(self, x, sliceDim):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            probability (torch.autograd.Variable): probability of x.

        """
        z = self._inferenceWithSlice(x, sliceDim, True)
        return self.prior.logProbability(z) + self._inferenceLogjac

    def _logProbabilityWithContraction(self, x, mask, mask_, sliceDim):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
            mask_ (torch.Tensor): mask_ = 1-mask.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            probability (torch.autograd.Variable): probability of x.

        """
        z = self._inferenceWithContraction(x, mask, mask_, sliceDim, True)
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

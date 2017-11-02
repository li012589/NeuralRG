import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    """

    def __init__(self, shapeList, sList, tList, prior, name=None):
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

        self.tList = torch.nn.ModuleList(tList)
        self.tNumLayers = len(self.tList)

        self.sList = torch.nn.ModuleList(sList)
        self.sNumLayers = len(self.sList)
        assert self.sNumLayers == self.tNumLayers

        self.NumLayers = self.sNumLayers
        self.prior = prior
        self.shapeList = shapeList
        if name is None:
            self.name = "realNVP_" + \
                str(self.sNumLayers) + "inner_" + \
                "_layers_" + self.prior.name + "Prior"
        else:
            self.name = name
        self._logjac = None

    def _generate(self, y, mask, mask_, ifLogjac=False):
        """

        This method generate complex distribution using variables sampled from prior distribution.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
        Return:
            y (torch.autograd.Variable): output Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.

        """
        if ifLogjac:
            self._generateLogjac = Variable(torch.zeros(y.data.shape[0]))
        for i in range(self.sNumLayers):
            if i % 2 == 0:
                y_ = mask * y
                s = self.sList[i](y_)*mask_
                t = self.tList[i](y_)*mask_
                y = y_ + mask_ * (y * torch.exp(s) + t)
                if ifLogjac:
                    for i in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
            else:
                y_ = mask_ * y
                s = self.sList[i](y_)*mask
                t = self.tList[i](y_)*mask
                y = y_ + mask * (y * torch.exp(s) + t)
                if ifLogjac:
                    for i in self.shapeList:
                        s = s.sum(dim=-1)
                    self._generateLogjac += s
        return y, mask,mask_

    def _inference(self, y, mask,mask_,ifLogjac=False):
        """

        This method inference prior distribution using variable sampled from complex distribution.
        Args:
            y (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
        Return:
            y (torch.autograd.Variable): output Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.

        """
        if ifLogjac:
            self._inferenceLogjac = Variable(torch.zeros(y.data.shape[0]))
        for i in list(range(self.sNumLayers))[::-1]:
            if (i % 2 == 0):
                y_ = mask * y
                s = self.sList[i](y_)*mask_
                t = self.tList[i](y_)*mask_
                y = mask_ * (y - t) * torch.exp(-s) + y_
                if ifLogjac:
                    for i in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
            else:
                y_ = mask_ * y
                s = self.sList[i](y_)*mask
                t = self.tList[i](y_)*mask
                y = mask * (y - t) * torch.exp(-s) + y_
                if ifLogjac:
                    for i in self.shapeList:
                        s = s.sum(dim=-1)
                    self._inferenceLogjac -= s
        return y, mask,mask_

    def _logProbability(self, x, mask,mask_):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            mask (torch.Tensor): mask to divide y into y0 and y1.
        Return:
            probability (torch.autograd.Variable): probability of x.

        """
        z, _,_ = self._generate(x, mask,mask_,True)
        return self.prior.logProbability(z) + self._generateLogjac

    def _saveModel(self, saveDic):
        """

        This methods add contents to saveDic, which will be saved outside.
        Args:
            saveDic (dictionary): contents to save.
        Return:
            saveDic (dictionary): contents to save with nerual networks in this class.

        """
        # save is done some where else, adding s,t to the dict
        for i in range(self.sNumLayers):
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
        for i in range(self.sNumLayers):
            self.sList[i].load_state_dict(saveDic["__" + str(i) + 'sLayer'])
            self.tList[i].load_state_dict(saveDic["__" + str(i) + 'tLayer'])
        return saveDic


class PriorTemplate():
    """

    This is the template class for prior, which will be used in realNVP class.
    Args:
        name (PriorTemplate): name of this prior.

    """

    def __init__(self, name="prior"):
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

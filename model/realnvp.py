import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model import RealNVPtemplate, PriorTemplate


class Gaussian(PriorTemplate):
    """

    This is a class for Gaussian prior distribution.
    Args:
        name (PriorTemplate): name of this prior.
        shapeList (int list): shape of sampled variables.

    """

    def __init__(self, shapeList, name="gaussian"):
        """

        This method initialise this class.
        Args:
            shapeList (int list): shape of sampled variables.
            name (PriorTemplate): name of this prior.

        """
        super(Gaussian, self).__init__(name)
        self.shapeList = shapeList

    def __call__(self, batchSize, volatile=False):
        """

        This method gives variables sampled from prior distribution.
        Args:
            batchSize (int): size of batch of variables to sample.
            volatile (bool): if only want forward, flag volatile to True to disable computation graph.
        Return:
            Samples (torch.autograd.Variable): sampled variables.

        """
        size = [batchSize] + self.shapeList
        return Variable(torch.randn(size), volatile=volatile)

    def logProbability(self, z):
        """

        This method gives the log probability of z in prior distribution.
        Args:
            z (torch.autograd.Variable): variables to get log probability of.
        Return:
            logProbability (torch.autograd.Variable): log probability of input variables.

        """
        tmp = -0.5 * (z**2)
        for i in self.shapeList:
            tmp = tmp.sum(dim=-1)
        return tmp


class MLP(nn.Module):
    """

    This is a class for multilayer perceptron.
    Args:
        fc* (nn.Linear): fully connected layers.
        name (string): name for this class.

    """

    def __init__(self, inNum, hideNum, name="mlp"):
        """

        This mehtod initialise this class.
        Args:
            inNum (int): number of elements in input variables.
            hideNum (int): number of elements in hide layer.
            name (string): name of this class.

        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inNum, hideNum)
        self.fc2 = nn.Linear(hideNum, inNum)
        self.name = name

    def forward(self, x):
        """

        This mehtod calculate the nerual network output of input x.
        Args:
            x (torch.autograd.Variable): input variables.
        Return:
            x (torch.autograd.Variable): output variables.

        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """

    This is a class for convolutional nerual network.
    Args:
        variableList (nn.ModuleList): list of series of convolutional nerual network layers.
        name (string): name for this class.

    """

    def __init__(self, inShape, netStructure, name="cnn"):
        """

        This mehtod initialise this class.
        Args:
            inShape (list int): shape of input variables.
            netStructure (list int): parameters of inner cnn layers, each items has 4 integer, which is channels of outputs, filter size, stride, padding in sequence.
            name (string): name of this class.

        """
        super(CNN, self).__init__()
        self.variableList = nn.ModuleList()
        former = inShape[0]
        self.name = name
        for layer in netStructure[:-1]:
            self.variableList.append(nn.Sequential(
                nn.Conv2d(former, layer[0], layer[1], layer[2], layer[3]), nn.ReLU()))
            former = layer[0]
        layer = netStructure[-1]
        self.variableList.append(nn.Sequential(
            nn.Conv2d(former, layer[0], layer[1], layer[2], layer[3])))
        #assert layer[0] == inshape[0]

    def forward(self, x):
        """

        This mehtod calculate the nerual network output of input x.
        Args:
            x (torch.autograd.Variable): input variables.
        Return:
            x (torch.autograd.Variable): output variables.

        """
        for layer in self.variableList:
            x = layer(x)
        return x


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

    def __init__(self, shapeList, sList, tList, prior, maskTpye="channel", name=None):
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
            shapeList, sList, tList, prior, name=name)
        self.createMask("channel")

    def createMask(self, maskType="channel", ifByte=1):
        """

        This method create mask for x, and save it in self.mask for later use.
        Args:
            maskType (string): specify waht type of mask to create. "channel" or "checkerboard".
            ifByte (int): flag variable, tell if output variable should be ByteTensor or FloatTensor.
        Return:
            mask (torch.Tensor): mask to divide x into y0 and y1.

        """
        size = self.shapeList.copy()
        if maskType == "channel":
            size[0] = size[0] // 2
            maskOne = torch.ones(size)
            maskZero = torch.zeros(size)
            mask = torch.cat([maskOne, maskZero], 0)
            self.mask = Variable(mask)
            self.mask_ = Variable(1 - mask)
        elif maskType == "checkerboard":
            assert (size[1] % 2 == 0)
            assert (size[2] % 2 == 0)
            unit = torch.Tensor([[1, 0], [0, 1]])
            self.mask = Variable(unit.repeat(
                size[0], size[1] // 2, size[2] // 2))
            self.mask_ = (1 - self.mask)
        else:
            raise ValueError("maskType not known.")
        if ifByte:
            self.mask = self.mask.byte()
            self.mask_ = self.mask_.byte()
        if self.ifCuda:
            self.mask = self.mask.cuda()
            self.mask_ = self.mask_.cuda()
        return self.mask

    def cuda(self):
        """

        This method move everything in RealNVP to GPU.
        Return:
            cudaModel (nn.Module.cuda): the instance in GPU.

        """
        cudaModel = super(RealNVP, self).cuda()
        if cudaModel.mask is not None:
            cudaModel.mask = self.mask.cuda()
            cudaModel.mask_ = self.mask_.cuda()
        return cudaModel

    def cpu(self):
        """

        This method move everything in RealNVP to CPU.
        Return:
            cudaModel (nn.Module): the instance in CPU.

        """
        cpuModel = super(RealNVP, self).cpu()
        if cpuModel.mask is not None:
            cpuModel.mask = self.mask.cpu()
            cpuModel_.mask = self.mask_.cpu()
        return cpuModel

    def generate(self, z, sliceDim=0):
        """

        This method generate complex distribution using variables sampled from prior distribution.
        Args:
            z (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            x (torch.autograd.Variable): output Variable.

        """
        return self._generateWithContraction(z, self.mask, self.mask_, sliceDim)

    def inference(self, x, sliceDim=0):
        """

        This method inference prior distribution using variable sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            z (torch.autograd.Variable): output Variable.

        """
        return self._inferenceWithContraction(x, self.mask, self.mask_, sliceDim)

    def logProbability(self, x, sliceDim=0):
        """

        This method gives the log of probability of x sampled from complex distribution.
        Args:
            x (torch.autograd.Variable): input Variable.
            sliceDim (int): in which dimension should mask be used on y.
        Return:
            log-probability (torch.autograd.Variable): log-probability of x.

        """
        return self._logProbabilityWithContraction(x, self.mask, self.mask_, sliceDim)

    def saveModel(self, saveDic):
        """

        This methods add contents to saveDic, which will be saved outside.
        Args:
            saveDic (dictionary): contents to save.
        Return:
            saveDic (dictionary): contents to save with nerual networks in this class.

        """
        self._saveModel(saveDic)
        saveDic["mask"] = self.mask  # Do check if exist !!
        saveDic["mask_"] = self.mask_
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
        self.mask = saveDic["mask"]
        self.mask_ = saveDic["mask_"]
        self.shapeList = saveDic["shapeList"]
        return saveDic

    def sample(self, batchSize, sliceDim=0, useGenerate=True):
        """

        This method directly sample samples of batch size given
        Args:
            batchSize (int): size of sampled batch.
            sliceDim (int): in which dimension should mask be used on y.
        return:
            samples: (torch.autograd.Variable): output Variable.
        """
        if self.ifCuda:
            z = self.prior(batchSize).cuda()
        else:
            z = self.prior(batchSize)
        if useGenerate:
            return self.generate(z, sliceDim)
        else:
            return self.inference(z, sliceDim)

    def __call__(self, batchSize, sliceDim=0, useGenerate=True):
        """
        This method is a wrapped sample method
        Args:
            batchSize (int): size of sampled batch.
            sliceDim (int): in which dimension should mask be used on y.
        return:
            samples: (torch.autograd.Variable): output Variable.
        """
        return self.sample(batchSize, sliceDim, useGenerate)


if __name__ == "__main__":

    pass

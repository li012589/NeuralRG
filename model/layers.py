import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class scalableTanh(nn.Module):
    def __init__(self,inNum):
        super(scalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.randn(inNum))
    def forward(self,input):
        input = F.tanh(input)
        input = input*self.scale
        return input

class MLP(nn.Module):
    """

    This is a class for multilayer perceptron.
    Args:
        fc* (nn.Linear): fully connected layers.
        name (string): name for this class.

    """

    def __init__(self, inNum, hideNum, name="mlp", activation = F.tanh):
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
        self.activation = activation

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
        x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self,dList, name="FC", activation=None):
        super(FC,self).__init__()
        if activation is None:
            activation = [nn.ReLU for _ in range(len(dList)-1)]
            activation.append(F.tanh)
        self.activation = activation
        assert(len(dList) == len(activation))
        fcList = []
        self.name = name
        self.activation = activation
        for i, d in enumerate(dList):
            if i == 0:
                pass
            else:
                fcList.append(nn.Linear(dList[i-1],dList[i]))
        self.fcList = torch.nn.ModuleList(fcList)
    def forward(self,x):
        tmp = x
        for i,layer in enumerate(self.fcList):
            tmp = layer(tmp)
            tmp = self.activation[i](tmp)
        return tmp

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
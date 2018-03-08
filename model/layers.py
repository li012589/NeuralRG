import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ScalableTanh(nn.Module):
    def __init__(self,input_size):
        super(ScalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.zeros(input_size))
    def forward(self,x):
        return self.scale * F.tanh(x)

class MLP(nn.Module):
    """

    This is a class for multilayer perceptron.
    Args:
        fc* (nn.Linear): fully connected layers.
        name (string): name for this class.

    """

    def __init__(self, inNum, hideNum, activation=None, name="mlp"):
        """

        This mehtod initialise this class.
        Args:
            inNum (int): number of elements in input variables.
            hideNum (int): number of elements in hide layer.
            name (string): name of this class.

        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inNum, hideNum)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(hideNum, inNum)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0)
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
        x = F.elu(x)
        x = self.fc2(x)
        if self.activation is None:
            return x
        else:
            x = self.activation(x)
            return x

class CNN(nn.Module):
    """

    This is a class for convolutional nerual network.
    Args:
        variableList (nn.ModuleList): list of series of convolutional nerual network layers.
        name (string): name for this class.

    """

    def __init__(self, netStructure,inchannel = 1 ,activation=None,name="cnn"):
        """

        This mehtod initialise this class.
        Args:
            netStructure (list int): parameters of inner cnn layers, each items has 4 integer, which is channels of outputs, filter size, stride, padding in sequence.
            name (string): name of this class.

        """
        super(CNN, self).__init__()
        self.variableList = nn.ModuleList()
        self.name = name
        for layer in netStructure[:-1]:
            self.variableList.append(nn.Sequential(
                nn.Conv2d(inchannel, layer[0], layer[1], layer[2], layer[3]), 
                #nn.MaxPool2d(layer[1], layer[2], layer[3]), 
                nn.ELU()))
            inchannel = layer[0]
        layer = netStructure[-1]
        self.variableList.append(nn.Sequential(
                nn.Conv2d(inchannel, layer[0], layer[1], layer[2], layer[3]),
                #nn.MaxPool2d(layer[1], layer[2], layer[3]), 
                ))
        self.activation = activation
        
        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

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

        if self.activation is not None:
            return self.activation(x)
        else:
            return x 


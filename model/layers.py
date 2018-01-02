import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Roll(nn.Module):
    def __init__(self,step,axis):
        super(Roll,self).__init__()
        if not isinstance(step,list):
            assert not isinstance(axis,list)
            step = [step]
            axis = [axis]
        assert len(step) == len(axis)
        self.step = step
        self.axis = axis
    def forward(self,x):
        shape = x.shape
        for i,s in enumerate(self.step):
            if s >=0:
                x1 = x.narrow(self.axis[i],0,s)
                x2 = x.narrow(self.axis[i],s,shape[self.axis[i]]-s)
            else:
                x2 = x.narrow(self.axis[i],shape[self.axis[i]]+s,-s)
                x1 = x.narrow(self.axis[i],0,shape[self.axis[i]]+s)
            x = torch.cat([x2,x1],self.axis[i])
        return x

class Wide2bacth(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        pass

class ScalableTanh(nn.Module):
    def __init__(self,input_size):
        super(ScalableTanh,self).__init__()
        self.scale = nn.Parameter(torch.zeros(input_size))
    def forward(self,x):
        return self.scale * F.tanh(x)


class Squeezing(nn.Module):
    def __init__(self,filterSize = 2):
        super(Squeezing,self).__init__()
        self.filterSize = filterSize
    def forward(self,input):
        scale_factor = self.filterSize
        batch_size, in_channels, in_height, in_width = input.size()

        out_channels = int(in_channels // (scale_factor * scale_factor))
        out_height = int(in_height * scale_factor)
        out_width = int(in_width * scale_factor)

        if scale_factor >= 1:
            input_view = input.contiguous().view(
            batch_size, out_channels, scale_factor, scale_factor,
            in_height, in_width)
            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        else:
            block_size = int(1 / scale_factor)
            input_view = input.contiguous().view(
            batch_size, in_channels, out_height, block_size,
            out_width, block_size)
            shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.view(batch_size, out_channels, out_height, out_width)

class MLP(nn.Module):
    """

    This is a class for multilayer perceptron.
    Args:
        fc* (nn.Linear): fully connected layers.
        name (string): name for this class.

    """

    def __init__(self, inNum, hideNum, activation=F.tanh, name="mlp"):
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
        x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self,dList,activation=None,name="FC"):
        super(FC,self).__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(dList)-1)]
            activation.append(nn.Tanh())
        self.activation = activation
        assert(len(dList) == len(activation))
        fcList = []
        self.name = name
        for i, d in enumerate(dList):
            if i == 0:
                pass
            else:
                fcList.append(nn.Linear(dList[i-1],dList[i]))
                fcList.append(activation[i])
        self.fcList = torch.nn.ModuleList(fcList)
    def forward(self,x):
        tmp = x
        for layer in self.fcList:
            tmp = layer(tmp)
        return tmp

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


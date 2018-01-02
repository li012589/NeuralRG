import torch
import torch.nn as nn
from torch.autograd import Variable 

class TEBD(nn.Module):
    '''
    the user interface is identical to RealNVP
    '''

    def __init__(self, prior, layers, name=None):
        super(TEBD,self).__init__()
        self.prior = prior 
        self.layers = torch.nn.ModuleList(layers) # a list of RNVP each take 2 in 2 out
        self.nlayers = len(self.layers)
        self.name = name

    def inference(self,x, ifLogjac=False):
        '''
        perform inference from x to z
        '''

        batchsize = x.data.shape[0]
        nbit  = x.data.shape[1]
        assert(nbit%2==0)

        if ifLogjac:
            self._inferenceLogjac = Variable(torch.zeros(batchsize))
        
        for l in list(range(self.nlayers))[::-1]:
            if (l%2==0): # even layer

                xout = Variable(torch.zeros(batchsize, nbit))
                for b in range(nbit//2): 
                    xin = Variable(torch.zeros(batchsize, 2))
                    xin = x[:, b*2:(b+1)*2]
                    xout[:, b*2:(b+1)*2] = self.layers[l].inference(xin, ifLogjac=ifLogjac)
                    if ifLogjac:
                        self._inferenceLogjac += self.layers[l]._inferenceLogjac
                x = xout

            else:        # odd layer

                xout = Variable(torch.zeros(batchsize, nbit))
                for b in range(nbit//2): 
                    xin = Variable(torch.zeros(batchsize, 2))
                    #wrap around
                    if (b<nbit//2-1):
                        xin = x[:, b*2+1:(b+1)*2+1]
                    else:
                        xin[:, 0] = x[:, nbit-1]
                        xin[:, 1] = x[:, 0]
                    xout[:, b*2:(b+1)*2] = self.layers[l].inference(xin,ifLogjac=ifLogjac)
                    if ifLogjac:
                        self._inferenceLogjac += self.layers[l]._inferenceLogjac

                y = Variable(torch.zeros(batchsize, nbit))
                for i in range(nbit):
                    y[:, (i+1)%nbit] = xout[:, i]
                x = y 
        return x 

    def generate(self, x, ifLogjac=False):

        '''
        inverse mapping of inference
        '''
        batchsize = x.data.shape[0]
        nbit  = x.data.shape[1]
        assert(nbit%2==0)

        if ifLogjac:
            self._generateLogjac = Variable(torch.zeros(batchsize))
        
        for l, layer in enumerate(self.layers):
            if (l%2==0): # even layer
                xout = Variable(torch.zeros(batchsize, nbit))
                for b in range(nbit//2):
                    xin = Variable(torch.zeros(batchsize, 2))
                    xin = x[:, b*2:(b+1)*2]
                    xout[:, b*2:(b+1)*2] = layer.generate(xin, ifLogjac=ifLogjac)
                    if ifLogjac:
                        self._generateLogjac += layer._generateLogjac
                x = xout
            else:        # odd layer

                xout = Variable(torch.zeros(batchsize, nbit))
                for b in range(nbit//2): 
                    xin = Variable(torch.zeros(batchsize, 2))
                    #wrap around
                    if (b<nbit//2-1):
                        xin = x[:, b*2+1:(b+1)*2+1]
                    else:
                        xin[:, 0] = x[:, nbit-1]
                        xin[:, 1] = x[:, 0]

                    xout[:, b*2:(b+1)*2]= layer.generate(xin,ifLogjac=ifLogjac)
                    if ifLogjac:
                        self._generateLogjac += layer._generateLogjac

                y = Variable(torch.zeros(batchsize, nbit))
                for i in range(nbit):
                    y[:, (i+1)%nbit] = xout[:, i]
                x = y 

        return x

    
    def logProbability(self, x):
        z = self.inference(x, True)
        return self.prior.logProbability(z) + self._inferenceLogjac 

    def sample(self, batchsize):
        z = self.prior(batchsize)
        return self.generate(z)


    def saveModel(self, saveDic):
        'should recursively call saveModel of all RNVP blocks'
        pass 
        #for layer in self.layers: 
        #    layer.saveModel(saveDic)


if __name__=='__main__':
    from model import Gaussian, MLP, RealNVP

    #RNVP block
    Nlayers = 4 
    Hs = 10 
    Ht = 10 
    sList = [MLP(2, Hs) for _ in range(Nlayers)]
    tList = [MLP(2, Ht) for _ in range(Nlayers)]
    masktypelist = ['channel', 'channel'] * (Nlayers//2)
    
    #assamble RNVP blocks into a TEBD layer
    prior = Gaussian([8])
    layers = [RealNVP([2], 
                      sList, 
                      tList, 
                      Gaussian([2]), 
                      masktypelist) for _ in range(4)] 
    
    tebd = TEBD(prior, layers)

    x = tebd.sample(10)
    print (x)

    z = tebd.inference(x)
    print (z)
    print (tebd.generate(z))

    #logp = tebd.logProbability(x)

    #print (logp)

    #params = list(tebd.parameters()) 
    #print (params)

    

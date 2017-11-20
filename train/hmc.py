import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.autograd import Variable
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptanceRate
from numpy.testing import assert_array_almost_equal,assert_array_equal

def metropolis(e1,e2):
    diff = e1-e2
    return diff.exp()-diff.uniform_()>=0.0

class HMCSampler:
    def __init__(self,model,prior,collectdata,stepSize=0.1,interSteps=10,targetAcceptRate=0.65,stepSizeMin=0.001,stepSizeMax=1000,stepSizeChangeRatio=0.03,dynamicStepSize=False):
        pass
        self.model = model
        self.prior = prior
        self.stepSize = stepSize
        self.interSteps = interSteps
        self.collectdata = collectdata
        self.dynamicStepSize = dynamicStepSize
        if self.dynamicStepSize:
            self.targetAcceptRate = targetAcceptRate
            self.stepSizeMax = stepSizeMax
            self.stepSizeMin = stepSizeMin
            self.stepSizeInc = stepSizeChangeRatio+1
            self.stepSizeDec = 1-stepSizeChangeRatio

    def updateStepSize(self,accept,stepSize):
        ratio = torch.sum(accept)/accept.shape[0]*accept[1]
        if ratio > self.targetAcceptRate:
            newStepSize = stepSize*self.stepSizeInc
        else:
            newStepSize = stepSize*self.stepSizeDec
        newStepSize = max(min(self.stepSizeMax,newStepSize),self.stepSizeMin)
        return newStepSize

    @staticmethod
    def hmcUpdate(z,v,model,stepSize,interSteps):
        force = model.backward(z)
        vp = v - 0.5*stepSize*force
        zp  = z + stepSize*vp
        for i in range(interSteps):
            force = model.backward(zp)
            vp -= stepSize*force
            zp += stepSize*vp
        force = model.backward(zp)
        vp = vp - 0.5*stepSize*force
        return zp,vp

    @staticmethod
    def hamiltonian(energy,v):
        return energy+0.5*torch.sum(v**2,1)

    def step(self,z):
        '''
        if isinstance(z,torch.DoubleTensor):
            v = torch.randn(z.size()).double()
        else:
            v = torch.randn(z.size())
        '''
        if isinstance(z,torch.DoubleTensor):
            v = torch.randn(z.size()).double()
        else:
            v = torch.randn(z.size())
        #x = z.clone()
        zp,vp = self.hmcUpdate(z,v,self.model,self.stepSize,self.interSteps)
        accept = metropolis(self.hamiltonian(self.model(z),v),self.hamiltonian(self.model(zp),vp))
        if self.dynamicStepSize:
            self.stepSize = self.updateStepSize(accept,self.stepSize)
        #print(type(accept.numpy()))
        accept = np.array([accept.numpy()]*self.model.nvars).transpose()
        mask = 1-accept
        x = torch.from_numpy(z.numpy()*mask +zp.numpy()*accept)
        accratio = accept.mean()
        #accept = accept.view(-1,1)
        #x.masked_scatter_(accept, torch.masked_select(z, accept))
        #assert_array_almost_equal(x.cpu().numpy(),z.cpu().numpy())
        return accratio,x

    def run(self,batchSize,ntherm,nmeasure,nskip):
        z = self.prior(batchSize).data
        for _ in range(ntherm):
            _,z = self.step(z)

        zpack = []
        measurePack = []
        accratio = 0.0
        for _ in range(nmeasure):
            for _ in range(nskip):
                a,z = self.step(z)
                accratio += a
            if self.collectdata:
                z_ = z.cpu().numpy()
                logp = self.model(z).cpu().numpy()
                logp.shape = (-1, 1)
                zpack.append(np.concatenate((z_, logp), axis=1))
                #zpack.append(z_)
            measure = self.measure(z)
            measurePack.append(measure)
        accratio /= float(nmeasure * nskip)
        print ('#accratio:', accratio)
        return zpack,measurePack,accratio

    def measure(self, x,measureFn=None):
        """
        This method measures some varibales.
        Args:
            measureFn (function): function to measure variables. If None, will run measure at target class.
        """
        if measureFn is None:
            measurements = self.model.measure(x)
        else:
            measurements = measureFn(x)
        return measurements

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

    #from train.objectives import Phi42 as Phi4
    from train.objectives import Phi4
    from model import Gaussian
    modelSize =216

    kappalist = np.arange(0.15,0.22,0.01).tolist()
    lamb = 1.145

    bins = 2
    dims = 3
    l = 6
    BatchSize = 100

    gaussian = Gaussian([modelSize])

    res = []
    errors = []
    for kappa in kappalist:

        model = Phi4(l, dims, kappa, lamb)
        sampler = HMCSampler(model,gaussian,True,dynamicStepSize=True)
        ret,mres,_ = sampler.run(BatchSize,300,500,1)
        ret= np.array(ret)
        z_o = ret[:,:,:-1]
        #z_ = np.reshape(ret,[-1,modelSize])
        m_abs = np.mean(z_o,2)
        m_abs = np.absolute(m_abs)
        m_abs_p = np.mean(m_abs)
        autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
        acceptRate = acceptanceRate(z_o)
        print("kappa:",model.kappa)
        print("lambda:",model.lamb)
        res.append(m_abs_p)
        errors.append(error)
        print("measure: <|m|/V>",m_abs_p,"with error:",error)
        print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

    print("Results: ",res)
    print("Errors: ",errors)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.errorbar(kappalist,res,yerr=errors)
    ax.set_title("\langle$|m|/V,\lambda = 1.145\rangle$")
    ax.set_ylabel("$\langle|m|/V\rangle$")
    ax.set_xlabel("$\kappa$")
    plt.show()

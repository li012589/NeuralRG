import flow
import source
import math
from flow import Flow

class OnebyonePlusRNVP(Flow):
    def __init__(self,maskList, tList, sList, h,w,c, prior=None, name="OnebyonePlusRNVP"):
        super(OnebyonePlusRNVP,self).__init__(prior,name)
        self.onebyone = flow.OnebyoneConv(h,w,c)
        self.rnvp = flow.RNVP(maskList, tList, sList)
    def inverse(self,y):
        y,inverseLogjac = self.onebyone.inverse(y)
        y,inverseLogjac1 = self.rnvp.inverse(y)
        inverseLogjac += inverseLogjac1
        return y,inverseLogjac
    def forward(self,z):
        z,forwardLogjac = self.rnvp.forward(z)
        z,forwardLogjac1 = self.onebyone.forward(z)
        forwardLogjac += forwardLogjac1
        return z,forwardLogjac

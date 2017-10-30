from model import RealNVPtemplate,LayerTemplate

class VanillaRealNVP(RealNVPtemplate):
    def __init__(self,numVars,layers,name = "vanillaRealNVP"):
        super(RealNVP,self).__init__(layers,name)
        self.numVars = numVars
        self.halfNumVars = int(numVars/2)
        maskOne = torch.ones(self.halfNumVars)
        maskZero = torch.zeros(self.halfNumVars)
        self.mask = torch.cat(tmp,tmp_,0)

class CouplingLayer(LayerTemplate):
    def __init__(self,s,t,name="CouplingLayer"):
        super(CouplingLayer,self).__init__(s,t,name)


if __name__ == "__main__":
    model = VanillaRealNVP()
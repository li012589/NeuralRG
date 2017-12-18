import scipy.sparse as sps 
from numpy import zeros 

class Lattice:
    def __init__(self,L, d, BC='periodic'):
        self.L = L 
        self.d = d
        self.shape = [L]*d 
        self.Nsite = L**d 
        self.BC = BC

    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift

        if self.BC != 'periodic':
            if (coord[d]>=self.L) or (coord[d]<0):
                return None
        
        #wrap around because of the PBC
        if (coord[d]>=self.L): coord[d] -= self.L; 
        if (coord[d]<0): coord[d] += self.L; 

        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = zeros(self.d, int) 
        for d in range(self.d):
            coord[self.d-d-1] = idx%self.L;
            idx /= self.L
        return coord 

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.d):
            idx *= self.L; 
            idx += coord[d]
        return idx 

class Hypercube(Lattice):
    def __init__(self,L, d, BC='periodic'):
        super(Hypercube, self).__init__(L, d, BC)
        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0
    
class Triangular(Lattice):
    def __init__(self, L):
        super(Triangular, self).__init__(L, 2)
        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                if j is not None:
                    self.Adj[i, j] = 1.0
                    self.Adj[j, i] = 1.0
            
            #diagonal 
            j = self.move(i, 0, 1)
            k = self.move(j, 1, 1)

            if (j is not None) and (k is not None):
                self.Adj[i, k] = 1.0
                self.Adj[k, i] = 1.0

if __name__=='__main__':
    from scipy.linalg import eigh 
    L=4
    d=2
    lattice = Hypercube(L, d)
    #lattice = Triangular(L)
    print (lattice.Adj)
    sys.exit(1)

    w, v = eigh(lattice.Adj)    
    v.shape = (L, L, L**2)

    print (w)
    import matplotlib.pyplot as plt 
    plt.figure()
    #plt.subplots(211)
    #plt.imshow(v[:, :, -2])
    #plt.subplots(212)
    plt.imshow(v[:, :, 2], vmin=-1, vmax=1)
    plt.show()

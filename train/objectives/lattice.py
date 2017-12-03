import scipy.sparse as sps 
from numpy import zeros 

class Hypercube:
    def __init__(self,L, d):
        self.L = L 
        self.d = d
        self.Nsite = L**d 

        self.Adj = zeros((self.Nsite,self.Nsite), int)
        for i in range(self.Nsite):
            for d in range(self.d):
                j = self.move(i, d, 1)

                self.Adj[i, j] = 1.0
                self.Adj[j, i] = 1.0

        #self.Adj = sps.csr_matrix(Adj)
    
    def move(self, idx, d, shift):
        coord = self.index2coord(idx)
        coord[d] += shift
        
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

if __name__=='__main__':
    L=4
    d=2
    lattice = Hypercube(L, d)
    print (lattice.Adj)

import numpy as np

l = 2
layer = 1
dims = 2
batchSize = 1
step = 0.01
k = 0.15
lamb = 0
start = -2
end = 2
num = 20

def no2ij(n,dList):
    cood =  []
    d = len(dList)
    for dim in reversed(range(len(dList))):
        L = dList[dim]
        tmp1 = (int(n/(L**dim)))
        n -= tmp1*(L**dim)
        cood.append(tmp1)
    return cood

def ij2no(cood,dList):
    n = 0
    d = len(dList)
    for dim in reversed(range(len(dList))):
        L = dList[dim]
        n += (cood[d-dim-1])*(L**dim)
    return n

def Kijbuilder(dList,k,lamb,skip=[]):
    maxNo = 1
    for d in dList:
        maxNo *= d
    Kij = np.zeros([maxNo]*2)
    for no in range(maxNo):
        cood = no2ij(no,dList)
        for i in range(len(cood)):
            if i in skip:
                continue
            coodp = cood.copy()
            coodp[i] = (cood[i]+1)%dList[i]
            Kij[no,ij2no(coodp,dList)] += k
            coodp[i] = (cood[i]-1)%dList[i]
            Kij[no,ij2no(coodp,dList)] += k
    tmp = np.diag([lamb]*(maxNo))
    return Kij+tmp

def energy(Kij,x,lamb):
    batchSize = x.shape[0]
    out = (np.matmul(np.matmul(x.reshape(batchSize,1,-1),Kij),x.reshape(batchSize,-1,1))).reshape(-1)
    out += (((x.reshape(batchSize,-1)*x.reshape(batchSize,-1))-1)**2).sum(-1)*lamb
    out = np.exp(-out)
    return out

def fx(x):
    return x^2

def expandSpace(start,end,dims,num,endpoint=True):
    space = []
    for _ in range(dims):
        space.append(np.linspace(start,end,num,endpoint=endpoint))
    space = np.meshgrid(*space)
    space = np.concatenate(space,axis=0)
    space = space.reshape(dims,num**dims).transpose([1,0])
    return space

def main():
    Kij = Kijbuilder([layer]+[l]*dims,-k,1,skip=[0])
    ndim = layer
    for _ in range(dims):
        ndim *= l
    ran = np.linspace(start,end,100)
    j = ran[0]
    e = ran[-1]
    res = 0
    nsamples = 0
    for i in ran[1:]:
        if i != e:
            space = expandSpace(j,i,ndim,num,endpoint=False)
        else:
            space = expandSpace(j,i,ndim,num)
        en = energy(Kij,space,lamb)
        nsamples += en.shape[0]
        res += en.sum()
        j = i
        del space
    print(nsamples)
    res = res*(end-start)/nsamples
    print(res)
    print(np.log(res))


if __name__ == "__main__":
    main()

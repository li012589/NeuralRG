import torch
import numpy as np
import source
import flow
import utils
import train

import argparse
import h5py
import os
import glob
import math
import matplotlib.pyplot as plt
from flow import getIndeices,dispatch

parser = argparse.ArgumentParser(description='')
parser.add_argument("-folder",default='./opt/tmp/')
#parser.add_argument("-inverse",action='store_false', help="using inverse")
parser.add_argument("-h5",action='store_false', help="save data to h5 file")
parser.add_argument("-allv",action='store_false', help="save data to h5 file")
parser.add_argument("-show",action='store_false', help="show matplotlib plot")
parser.add_argument("-save",action='store_false', help="save matplotlib plot")
args = parser.parse_args()

with h5py.File(args.folder+"parameters.hdf5","r") as f:
    L = int(np.array(f["L"]))
    nrepeat = int(np.array(f["nrepeat"]))
    batch = int(np.array(f["batch"]))

with torch.no_grad():
    name = max(glob.iglob(args.folder+'records/*HMCresult*.hdf5'), key=os.path.getctime)
    print("load: "+name)
    with h5py.File(name,"r") as f:
        data = torch.from_numpy(np.array(f['data']))

    depth = int(math.log(L,2))

    RES = []
    result = utils.cor(data[0].view(batch,-1)).detach()
    RES.append(result)

    if args.allv:
        for l in range(depth+1):
            result = utils.cor(data[(l)*2*nrepeat+1].view(batch,-1))
            RES.append(result)

    else:
        shape = [L,L]
        kernelSize = 2
        indexList = []
        for no in range(depth):
            for _ in range(nrepeat):
                indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize*(kernelSize**no),kernelSize**no,0))
                indexList.append(getIndeices(shape,kernelSize,kernelSize,kernelSize*(kernelSize**no),kernelSize**no,kernelSize**no))
        indexIList = [item[0] for item in indexList]
        indexJList = [item[1] for item in indexList]
        for l in range(depth+1):
            tmp = data[l*2*nrepeat+1]
            if l >0:
                tmp,tmp_ = dispatch(indexIList[l*2*nrepeat-1],indexJList[l*2*nrepeat-1],tmp)
                tmp = tmp_
            result = utils.cor(tmp.view(batch,-1))
            RES.append(result)

t = torch.cat([t_.view(-1) for t_ in RES])
vmax = t.max().item()
vmin = t.min().item()
for i in RES:
    plt.matshow(i.detach().numpy(), cmap='Greys',vmax = vmax, vmin = vmin)
    plt.colorbar()

    plt.savefig(args.folder+"pic/correlation_original.pdf")

plt.show()

with h5py.File(args.folder+"pic/correlation.hdf5","w") as f:
    f.create_dataset("CORRE",data=np.array(RES))

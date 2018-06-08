import os
import sys
sys.path.append(os.getcwd())

import utils
import h5py
import torch
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from flow import getIndeices,dispatch

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-folder",default='./opt/tmp/')
parser.add_argument("-allv",action='store_false', help="save data to h5 file")
parser.add_argument("-show",action='store_false', help="show matplotlib plot")
parser.add_argument("-save",action='store_false', help="save matplotlib plot")
args = parser.parse_args()

rootFolder = args.folder
if rootFolder[-1] != '/':
    rootFolder += '/'
import pdb
pdb.set_trace()
with h5py.File(rootFolder+"parameters.hdf5","r") as f:
    L = int(np.array(f["L"]))
    nrepeat = int(np.array(f["nrepeat"]))
    batch = int(np.array(f["batch"]))
with torch.no_grad():
    name = max(glob.iglob(rootFolder+'records/*HMCresult*.hdf5'), key=os.path.getctime)
    print("load: "+name)
    with h5py.File(name,"r") as f:
        data = torch.from_numpy(np.array(f['data']))

    depth = int(math.log(L,2))

    RES = []
    result = utils.cor(data[0].view(batch,-1)).detach()
    RES.append(result.cpu().detach().numpy())

    if args.allv:
        for l in range(depth+1):
            result = utils.cor(data[(l)*2*nrepeat+1].view(batch,-1))
            RES.append(result.cpu().detach().numpy())

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
            RES.append(result.cpu().detach().numpy())

plt.figure(figsize=(8,12))
ax1 = plt.subplot(321)
plt.imshow(RES[1])
ax2 = plt.subplot(322)
plt.imshow(RES[2])
ax3 = plt.subplot(323)
plt.imshow(RES[3])
ax4 = plt.subplot(324)
plt.imshow(RES[4])
ax5 = plt.subplot(325)
plt.imshow(RES[5])



plt.show()

import time
import re
import numpy as np
import h5py

maximumJobs = 8

command = ['python','./replyMain.py','-epochs','5000','-batch','512','-nlayers','10','-nmlp','12','-nhidden','10','-L','16','-nrepeat','1','-savePeriod','100','-alpha','1']

settings = [['-cuda',str(i)] for i in range(8)]

parameters = {"-T":[str(i/10) for i in range(10,26)],"-depthMERA":[str(i+1) for i in range(4)][::-1]}

def before():
    #print("this is pre-process")
    pass

def after():
    #print("this is sub-process")
    pass

def finish(result):
    loss = []
    std = []
    for j in parameters['-depthMERA']:
        tmploss = []
        tmpstd = []
        for i in parameters['-T']:
            tmploss.append(result['-T '+i+' -depthMERA ' +j][-1][-2])
            tmpstd.append(result['-T '+i+' -depthMERA ' +j][-1][-1])
        loss.append(tmploss)
        std.append(tmpstd)
    print('loss:',loss)
    print('std:',std)

    loss = np.array(loss)
    std = np.array(std)
    with h5py.File("./core_result_Tfrom"+parameters['-T'][0]+"to"+parameters['-T'][-1]+"_depthMERAfrom"+parameters['-depthMERA'][0]+"to"+parameters['-depthMERA'][-1]+".hdf5","w") as f:
        f.create_dataset('loss',data = loss)
        f.create_dataset('std',data = std)

def process(result):
    nums = []
    for i in result[-2:-1]:
        nums.append([float(s) for s in re.findall(r'-?\d+\.?\d*',i)])
    return nums

if settings != []:
    assert len(settings) == maximumJobs

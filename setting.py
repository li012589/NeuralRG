import time
import re
import numpy as np
import h5py

maximumJobs = 7

command = ['python','./replyMain.py','-epochs','50','-batch','512','-nlayers','10','-nmlp','3','-nhidden','10','-L','32','-nrepeat','1','-savePeriod','100','-alpha','1']

settings = [['-cuda',str(i)] for i in range(7)]

parameters = {"-T":[str(i/10) for i in range(20,36)],"-depthMERA":[str(3),str(5)]}

def before():
    #print("this is pre-process")
    pass

def after():
    #print("this is sub-process")
    pass

def finish(result):
    res = []
    for j in parameters['-depthMERA']:
        tmp = []
        for i in parameters['-T']:
            tmp.append(result['-T '+str(i),+' -depthMERA ' +str(j)])
        res.append(tmp)
    print(res)
    res = np.array(res)
    with h5py.File("./core_result.hdf5","w") as f:
        f.create_dataset('result',res)

def process(result):
    nums = []
    for i in result[-2:-1]:
        nums.append([float(s) for s in re.findall(r'-?\d+\.?\d*',i)])
    return np.array(nums)

if settings != []:
    assert len(settings) == maximumJobs

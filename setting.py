import time
import re
import numpy as np

maximumJobs = 8

command = ['python','./replyMain.py','-epochs','5000','-batch','512','-nlayers','10','-nmlp','1','-nhidden','10','-L','32','-nrepeat','1','-savePeriod','100','-alpha','1']

settings = [['-cuda',str(i)] for i in range(8)]

parameters = {"-T":[str(i/10) for i in range(10,21)],"-depthMERA":[str(i+1) for i in range(4)]}

def before():
    #print("this is pre-process")
    pass

def after():
    #print("this is sub-process")
    pass

def process(result):
    nums = []
    for i in result[-2:-1]:
        nums.append([float(s) for s in re.findall(r'-?\d+\.?\d*',i)])
    return np.array(nums)

if settings != []:
    assert len(settings) == maximumJobs

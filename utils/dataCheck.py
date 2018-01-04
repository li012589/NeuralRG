import torch
import numpy as np

def checkNan(data):
    '''
    byte0 = (data == float('inf')).sum()
    assert(int(byte0.data[0]) == 0)
    byte1 = (data == -float('inf')).sum()
    assert(int(byte1.data[0]) == 0)
    byte2 = (data != data).sum()
    assert(int(byte2.data[0]) == 0)
    '''
    return data

import torch
import numpy as np

def checkNan(data):
    byte0 = (data == float('inf')).sum()
    assert(byte0 == 0)
    byte1 = (data == -float('inf')).sum()
    assert(byte1 == 0)
    byte2 = (data != data).sum()
    assert(byte1 == 0)

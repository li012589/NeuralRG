import torch
import numpy as np

def correlation(x):
    m = x.mean(0)
    xp = x - m
    c = xp.t().mm(xp)
    c = c/(x.shape[0]-1)
    std = torch.diag(c)**0.5
    s = std.expand_as(c)*std.expand_as(c).t()
    return c/s

def cor(x):
    batchSize = x.shape[0]
    return x.t().mm(x)/(batchSize)
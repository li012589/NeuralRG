import numpy as np
import torch

def getIndeices(shape,height,width,stride,dialation, offset):
    H, W = shape
    outHeight = (H - dialation*(height-1)-1) // stride +1
    outWidth = (W - dialation*(width-1)-1) // stride +1

    i0 = np.repeat(np.arange(height)*dialation, width)
    i1 = stride * np.repeat(np.arange(outHeight), outWidth)
    j0 = np.tile(np.arange(width)*dialation, height)
    j1 = stride * np.tile(np.arange(outWidth), outHeight)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (i.transpose(1,0)+offset)%H, (j.transpose(1,0)+offset)%W

def dispatch(i,j,x):
    x_ = x[:,:,i,j].reshape(x.shape[0],x.shape[1],i.shape[0],i.shape[1])
    return x, x_

def collect(i,j,x,x_):
    xi = x.clone()
    xi[:,:,i,j]=x_.reshape(x.shape[0],-1,i.shape[0],i.shape[1])
    return xi
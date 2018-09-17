import torch
from torch import nn
import numpy as np

import utils
import flow
import train
import source

import argparse

parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('Ising target parameters')
#
group.add_argument("-L",type=int, default=2,help="linear size")
group.add_argument("-d",type=int, default=2,help="dimension")
#group.add_argument("-T",type=float, default=2.269185314213022, help="Temperature")
group.add_argument("-kappa",type=float, default=0.15, help="Kappa")
group.add_argument("-lamb",type=float, default=1.145, help="Lambda")

group = parser.add_argument_group('integration parameters')
#
group.add_argument("-start",type=int, default=-2,help="start point")
group.add_argument("-end",type=int, default=2,help="end point")
group.add_argument("-double", action='store_true', help="use float64")
group.add_argument("-points",type=int, default=10,help="sampled points at each dimension")

args = parser.parse_args()

def expandSpace(start,end,dims,num,endpoint=True):
    space = []
    for _ in range(dims):
        space.append(np.linspace(start,end,num,endpoint=endpoint))
    space = np.meshgrid(*space)
    space = np.concatenate(space,axis=0)
    space = space.reshape(dims,num**dims).transpose([1,0])
    return torch.from_numpy(space)#.to(torch.float32)

with torch.no_grad():
    if args.double:
        t = source.Phi4(args.L,args.d,args.kappa,args.lamb).to(torch.float64)
        space = expandSpace(args.start,args.end,args.L**args.d,args.points)
    else:
        print("using float32")
        t = source.Phi4(args.L,args.d,args.kappa,args.lamb).to(torch.float32)
        space = expandSpace(args.start,args.end,args.L**args.d,args.points).to(torch.float32)
    volume = (args.end-args.start)**(args.L**args.d)
    enfn = lambda space,volume: torch.exp(-t.energy(space)).mean()*volume
    En = enfn(space,volume)
    print(En.item())
    print(-torch.log(En).item())
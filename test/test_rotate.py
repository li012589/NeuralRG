from flowRelated import *

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source


def test_bijective():
    p = source.Gaussian([2,2])

    f = flow.Rotate(prior=p)
    bijective(f)

def test_saveload():
    p = source.Gaussian([2,2])

    f = flow.Rotate(prior=p)
    p = source.Gaussian([2,2])
    blankf = flow.Rotate(prior=p)
    saveload(f,blankf)


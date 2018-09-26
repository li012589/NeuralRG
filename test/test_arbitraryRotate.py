from flowRelated import *
import pytest
import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np
import utils
import flow
import source

@pytest.mark.skip(reason="Known failure")
def test_bijective():
    p = source.Gaussian([3,2,2])
    f = flow.ArbitraryRotate(3,prior=p)
    bijective(f,decimal=3)
@pytest.mark.skip(reason="Known failure")
def test_saveload():
    p = source.Gaussian([3,2,2])
    f = flow.ArbitraryRotate(3,prior=p)
    blankf = flow.ArbitraryRotate(3,prior=p)
    saveload(f,blankf,decimal=3)

if __name__ == "__main__":
    test_bijective()
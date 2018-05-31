import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def logsumexp(logs):
    logs = torch.stack(logs,dim=0)
    m, _ = torch.max(logs, dim=0, keepdim=True)
    tmp = logs - m
    return m + torch.log(torch.sum(torch.exp(tmp),dim=0,keepdim=True))

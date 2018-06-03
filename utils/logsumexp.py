import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def logsumexp(logs):
    logs = torch.stack(logs,dim=1)
    return (logs-F.log_softmax(logs,dim=1)).mean(1)
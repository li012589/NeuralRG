import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class dataSet(Dataset):
    def __init__(self,path,dtype = np.float32):
        self.dataCore = np.loadtxt(path,dtype=dtype)

    def __len__(self):
        return self.dataCore.shape[0]

    def __getitem__(self,idx):
        return self.dataCore[idx,:]
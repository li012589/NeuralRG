import torch 
from torch.autograd import Variable 
import numpy as np 

from realnvp import RealNVP 

xy = np.loadtxt('train.dat', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, -1]))

print (x_data.data.shape)
print (y_data.data.shape)

Nvars = x_data.data.shape[-1]
print (Nvars)

model = RealNVP(Nvars)


criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):

    y_pred = model.logp(x_data)

    loss = criterion(y_pred, y_data)
    print (epoch, loss.data[0]) 

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()

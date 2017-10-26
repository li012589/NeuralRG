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

#after training, plot the samples draw from the realnvp net 
Nsamples = 1000
z = Variable(torch.randn(Nsamples, Nvars))
x = model.backward(z)

x = x.data.numpy()

import matplotlib.pyplot as plt 
plt.scatter(x[:,0], x[:,1], alpha=0.5, label='$x$')

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.ylabel('$x_1$')
plt.xlabel('$x_2$')
plt.legend() 


###########################
from generate_samples import test_logprob 
x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = np.exp( test_logprob([x[i], y[j]]) ) 
plt.contour(X, Y, Z)
###########################

plt.show()

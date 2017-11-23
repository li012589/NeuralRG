import numpy as np

def acceptanceRate(z):
    cnt = z.shape[0] * z.shape[1]
    for i in range(0, z.shape[1]):
        for j in range(1, z.shape[0]):
            if np.min(np.equal(z[j - 1,i], z[j,i])):
                cnt -= 1
    return cnt / float(z.shape[0] * z.shape[1])

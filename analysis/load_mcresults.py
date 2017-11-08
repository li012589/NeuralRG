import h5py 
import numpy as np 

import numpy as np

def binning_analysis(samples,bins=7):
    #Perform a binning analysis over samples and return an array of the error estimate at each binning level.
    minbins = 2**bins # minimum number of bins (128 still seems to be a reasonable sample size in most cases)
    maxlevel = int(np.log2(len(samples)/minbins))
    maxsamples = int(minbins * 2**(maxlevel))
    bins = np.array(samples[-maxsamples:]) # clip to power of 2 for simplicity
    errors = np.zeros(maxlevel+1)
    for k in range(maxlevel+1):
        errors[k] = np.std(bins)/np.sqrt(len(bins)-1.)
        bins = np.array([(bins[2*i]+bins[2*i+1])/2. for i in range(len(bins)//2)])
    error_naive = np.std(samples)/np.sqrt(len(samples)-1.)
    tau = .5*(errors[-1]**2/error_naive**2 - 1.)
    return errors[-1], tau

if __name__=='__main__':
    import argparse 

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-filename", help="filename")
    args = parser.parse_args()

    h5 = h5py.File(args.filename,'r')
    obs = np.array(h5['results']['obs'])
    h5.close()

    for ibatch in range(obs.shape[-1]):
        data = obs[:, ibatch]
        error, tau = binning_analysis(data) 
        print (data.mean(), error, tau)

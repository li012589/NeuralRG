import numpy as np

def binning_analysis(samples,bins):
    '''
    Calculate errors using binning method
    :param samples: [TimeStep,BatchSize,zSize], the data to be evaluated
    :param bins: int, minimum bin size = 2**bins
    :return errors: [binsNum,BatchSize], std errors
    :return mean_errors: [binsNum], = np.mean(errors,1)
    '''
    #Perform a binning analysis over samples and return an array of the error estimate at each binning level.
    minbins = 2**bins # minimum number of bins (128 still seems to be a reasonable sample size in most cases)
    maxlevel = int(np.log2(len(samples)/minbins))
    #print(maxlevel)
    maxsamples = int(minbins * 2**(maxlevel))
    bins = np.array(samples[-maxsamples:,:]) # clip to power of 2 for simplicity
    #errors = np.zeros(maxlevel+1)
    errors = []
    #print(bins)
    #print(bins.shape)
    for k in range(maxlevel+1):
        errors.append(np.std(bins,0)/np.sqrt(len(bins)-1.))
        #print("errors:")
        #print(errors)
        bins = np.array([(bins[2*i]+bins[2*i+1])/2. for i in range(len(bins)//2)])
        #print("binning")
        #print(bins)
        #print(bins.shape)
    errors = np.array(errors)
    #print("errors:")
    #print(errors)
    #mean_errors = np.mean(errors,1)
    #print(mean_errors)
    return errors

def auto_time(errors):
    #print(errors[-1,:])
    #print(errors[0,:])
    act = (.5*(errors[-1,:]**2/errors[0,:]**2 - 1.))
    #print("act:")
    #print(act)
    return np.mean(act),np.mean(errors[-1,:])

def autoCorrelationTime(samples,bins=7):
    errors = binning_analysis(samples,bins)
    tau,error =  auto_time(errors)
    return tau

def autoCorrelationTimewithErr(samples,bins=7):
    errors = binning_analysis(samples,bins)
    tau,error =  auto_time(errors)
    return tau,error
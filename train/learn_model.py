if __name__ =="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
import torch 
torch.manual_seed(42)
from torch.autograd import Variable 
import numpy as np 

from model import Gaussian,MLP,RealNVP
from train.objectives import Ring2D, Ring5 

def fit(Nlayers, Hs, Ht, Nepochs, supervised,ifCuda = False):
    LOSS=[]
    xy = np.loadtxt('train.dat', dtype=np.float32)
    x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
    if ifCuda:
        x_data = x_data.cuda()

    if supervised:
        y_data = Variable(torch.from_numpy(xy[:, -1]))
        if ifCuda:
            y_data = y_data.cuda()

    Nvars = x_data.data.shape[-1]

    gaussian = Gaussian([Nvars])

    sList = [MLP(Nvars//2, Hs) for i in range(Nlayers)] 
    tList = [MLP(Nvars//2, Ht) for i in range(Nlayers)] 

    model = RealNVP([Nvars], sList, tList, gaussian)
    model.createMask()
    if ifCuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    if supervised:
        criterion = torch.nn.MSELoss(size_average=True)

    for epoch in range(Nepochs):

        logp = model.logProbability(x_data)
        if supervised:
            loss = criterion(logp, y_data)
        else:
            loss = -logp.mean() 

        print (epoch, loss.data[0])
        LOSS.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

    saveDict = model.saveModel({})
    torch.save(saveDict, './'+model.name)
    return Nvars, x_data, model, LOSS

if __name__=="__main__":
    import h5py 
    import subprocess 
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-Nlayers", type=int, default=8, help="")
    parser.add_argument("-Hs", type=int, default=10, help="")
    parser.add_argument("-Ht", type=int, default=10, help="")
    parser.add_argument("-Nepochs", type=int, default=500, help="")
    parser.add_argument("-target", default='ring2d', help="target distribution")
    parser.add_argument("-cuda", action='store_true', help="use GPU")
    parser.add_argument("-folder", default='data/', help="where to store results")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-supervised", action='store_true', help="supervised")
    group.add_argument("-unsupervised", action='store_true', help="unsupervised")
    args = parser.parse_args()

    if args.target == 'ring2d':
        target = Ring2D()
    elif args.target == 'ring5':
        target = Ring5()
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    Nvars, x_data, model, LOSS= fit(args.Nlayers, 
                                    args.Hs, 
                                    args.Ht, 
                                    args.Nepochs, 
                                    args.supervised,
                                    args.cuda)

    #after training, generate some data from the network
    Ntest = 1000
    z = model.prior(Ntest, volatile=True) # prior 
    x = model.generate(z)  

    # on training data
    logp_model_train = model.logProbability(x_data)
    logp_data_train = target(x_data)

    # on test data
    logp_model_test = model.logProbability(x)
    logp_data_test = target(x)
    
    # store results
    cmd = ['mkdir', '-p', args.folder]
    subprocess.check_call(cmd)
    key = args.target \
         +'_Nl' + str(args.Nlayers) \
         +'_Hs' + str(args.Hs) \
         +'_Ht' + str(args.Ht) 

    if args.supervised:
        key += '_sl'
    else:
        key += '_ul'

    h5 = h5py.File(args.folder +'/'+key+'.h5','w')
    params = h5.create_group('params')
    params.create_dataset("Nvars", data=Nvars)
    params.create_dataset("Nlayers", data=args.Nlayers)
    params.create_dataset("Hs", data=args.Hs)
    params.create_dataset("Ht", data=args.Nlayers)
    params.create_dataset("target", data=args.target)
    params.create_dataset("supervised", data=args.supervised)
    params.create_dataset("unsupervised", data=args.unsupervised)

    results = h5.create_group('results')
    results.create_dataset("train_data",data=x_data.data.numpy())
    results.create_dataset("generated_data",data=x.data.numpy())
    results.create_dataset("logp_model_train",data=logp_model_train.data.numpy())
    results.create_dataset("logp_model_test",data=logp_model_test.data.numpy())
    results.create_dataset("logp_data_train",data=logp_data_train.data.numpy())
    results.create_dataset("logp_data_test",data=logp_data_test.data.numpy())
    results.create_dataset("loss",data=LOSS)

    h5.close()

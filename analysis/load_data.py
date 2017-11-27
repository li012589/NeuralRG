import h5py 
import matplotlib.pyplot as plt 
import numpy as np 

def figure1(h5):

    logp_model_train = np.array(h5['results']['logp_model_train'])
    #logp_model_test= np.array(h5['results']['logp_model_test'])
    logp_data_train = np.array(h5['results']['logp_data_train'])
    #logp_data_test = np.array(h5['results']['logp_data_test'])

    plt.figure()
    plt.scatter(logp_model_train, logp_data_train, alpha=0.5, label='training samples')
    #plt.scatter(logp_model_test, logp_data_test, alpha=0.5, label='generated samples')
    plt.xlabel('$\log{P(model)}$')
    plt.ylabel('$\log{P(baseline)}$')
    plt.legend()
    plt.title("$\log{P(x)}$")

def figure2(h5):

    supervised = int(h5['params']['supervised'][()])
    x_data = np.array(h5['results']['train_data'])
    x = np.array(h5['results']['generated_data'])

    plt.figure()
    plt.scatter(x_data[:,0], x_data[:,1], alpha=0.5, label='training samples')
    plt.scatter(x[:,0], x[:,1], alpha=0.5, label='generated samples')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    if supervised:
        plt.title("Supervised")
    else:
        plt.title("Unsupervised")
    plt.legend()


def figure3(h5):

    supervised = int(h5['params']['supervised'][()])
    loss = np.array(h5['results']['loss'])

    plt.figure()
    plt.plot(loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    if supervised:
        plt.title("MSE loss")
    else:
        plt.title("NLL loss")

if __name__=='__main__':
    import argparse 

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-filename", help="filename")
    parser.add_argument("-index", type=int, default=1, help="figure index")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-show", action='store_true',  help="show figure right now")
    group.add_argument("-outname", default="result.pdf",  help="output pdf file")
    args = parser.parse_args()

    h5 = h5py.File(args.filename,'r')
    
    #TODO: better selection 
    if args.index==1:
        figure1(h5)
    elif args.index==2:
        figure2(h5)
    elif args.index==3:
        figure3(h5)

    h5.close()

    if args.show:
        plt.show()
    else:
        plt.savefig(args.outname, dpi=300, transparent=True)

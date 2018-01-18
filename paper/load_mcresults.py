import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from config import * 
from matplotlib.offsetbox import AnchoredText

def loss_and_acc(h5):

    loss = np.array(h5['results']['loss'])

    plt.figure(figsize=(8,6))
    ax1 = plt.subplot(211)
    plt.plot(loss[:, 1], lw=2)
    ax1.get_xaxis().set_visible(False)
    at = AnchoredText("(a)",prop=dict(size=18), frameon=False,loc=2,)
    plt.gca().add_artist(at)

    plt.ylabel('$\mathcal{L}$')
    plt.subplot(212, sharex=ax1)
    plt.plot(loss[:, 2], lw=2)
    at = AnchoredText("(b)",prop=dict(size=18), frameon=False,loc=2,)
    plt.gca().add_artist(at)

    plt.ylabel('acceptance rate')
    plt.xlabel('iterations')
    plt.subplots_adjust(hspace=0.15, bottom=0.15,left=0.15)
    plt.xlim([0,2000])
    #plt.legend()

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
        loss_and_acc(h5)
    elif args.index==2:
        pass
    elif args.index==3:
        pass 

    h5.close()

    if args.show:
        plt.show()
    else:
        plt.savefig(args.outname, dpi=300, transparent=True)

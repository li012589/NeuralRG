import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from config import * 
from matplotlib.offsetbox import AnchoredText

def loss_and_acc(h5, exact=None):

    loss = np.array(h5['results']['loss'])

    plt.figure(figsize=(8,6))
    ax1 = plt.subplot(211)
    plt.plot(loss[:, 1], lw=2)
    ax1.get_xaxis().set_visible(False)
    at = AnchoredText("(a)",prop=dict(size=18), frameon=False,loc=2,)
    plt.gca().add_artist(at)
    at = AnchoredText("(b)",prop=dict(size=18), frameon=False,loc=2,)
    plt.ylabel('$\mathcal{L}$')


    ###################
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    #axins = zoomed_inset_axes(ax1, 1.5,  loc=7)
    axins = plt.axes([0.45, 0.65, 0.4, 0.2])
    axins.semilogx(loss[:, 1], lw=2)
    if exact is not None:
        plt.gca().axhline(exact, color='r', lw=2)
    
    x1, x2 = 100, 2000
    y1, y2 = -150, -140
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #axins.xaxis.set_visible(False)
    #axins.yaxis.set_visible(False)
    mark_inset(ax1, axins, loc1=1, loc2=2, fc='none', ec='0.5')
    ###################

    plt.subplot(212, sharex=ax1)
    plt.plot(loss[:, 2], lw=2)
    plt.gca().add_artist(at)

    plt.ylabel('acceptance rate')
    plt.xlabel('iterations')
    plt.subplots_adjust(hspace=0.1, bottom=0.15,left=0.15)
    plt.xlim([0,2000])
    #plt.legend()

if __name__=='__main__':
    import argparse 

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-filename", help="filename")
    parser.add_argument("-index", type=int, default=1, help="figure index")
    parser.add_argument("-exact", type=float, default=None, help="exact lower bound")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-show", action='store_true',  help="show figure right now")
    group.add_argument("-outname", default="result.pdf",  help="output pdf file")
    args = parser.parse_args()

    h5 = h5py.File(args.filename,'r')
    
    #TODO: better selection 
    if args.index==1:
        loss_and_acc(h5, args.exact)
    elif args.index==2:
        pass
    elif args.index==3:
        pass 

    h5.close()

    if args.show:
        plt.show()
    else:
        plt.savefig(args.outname, dpi=300, transparent=True)

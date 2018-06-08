import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import matplotlib
from matplotlib import rcParams

#rcParams['text.usetex'] = True
#rcParams['font.serif'] = 'Computer Modern Roman'

rcParams['lines.linewidth'] = 1

rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16


rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20

rcParams['legend.fancybox'] = True


def loss_and_acc(h5, exact=None):

    loss = np.array(h5['LOSS'])

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
    parser.add_argument("-folder",default='./opt/tmp/')
    parser.add_argument("-exact", type=float, default=None, help="exact lower bound")

    parser.add_argument("-show", action='store_true',  help="show figure right now")
    parser.add_argument("-save", action='store_true',  help="save figure to a file")
    parser.add_argument("-outname", default="result",  help="output pdf file")
    args = parser.parse_args()

    rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'

    parameterFileName = rootFolder+"parameters.hdf5"
    name = max(glob.iglob(rootFolder+"records/*Record*.hdf5"),key = os.path.getctime)
    print("Plotting loss at:"+name)
    with h5py.File(name,'r') as f:
        loss_and_acc(f, args.exact)

    if args.show:
        plt.show()
    if args.save:
        plt.savefig(args.outname, dpi=300, transparent=True)

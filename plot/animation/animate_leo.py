import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

def animate(datafile, imgfile=None):
    '''
    Ant March!
    '''
    data = np.loadtxt(datafile, usecols=(0, 1))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    def update(i):
        tt = plt.title('Step = %s'%i)
        x, y = data[i, 0], data[i, 1]
        c = plt.scatter(x, y, alpha=0.5, color='C0')
        #c = plt.Circle((x,y),radius=0.1)
        #plt.gca().add_patch(c)
        return [c, tt]

    anim = animation.FuncAnimation(plt.gcf(), update,  
                                   frames=len(data), 
                                   interval=20,repeat=False,
                                   blit=False)
    if imgfile is None:
        plt.show()
    else:
        anim.save(imgfile,dpi=80,writer='imagemagick',fps=50)

if __name__ == '__main__':
    import sys
    datafile = sys.argv[1]
    if len(sys.argv)>2:
        imgfile = sys.argv[2]
    else:
        imgfile = None
    animate(datafile, imgfile)

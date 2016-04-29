import sys
sys.path.append("../../data")

from generate import get_data
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def show(fname):

    img_master = get_data(dataset="crtsimg", fname="912201400354121224.master.fits")

    plt.imshow(img_master, cmap='gray', norm=LogNorm())
    plt.colorbar()
    plt.title('Master')

    for i in xrange(1,5):
        plt.figure()
        img = get_data(dataset="crtsimg", fname="912201400354121224-000%i.arch.fits" % i)
        plt.imshow(img, cmap='gray', norm=LogNorm())
        plt.title('000%i' % i)
        plt.colorbar()
    
    plt.show()

if __name__ == "__main__":

    show("CSS_trans_ims/912201400044136465.master.fits")

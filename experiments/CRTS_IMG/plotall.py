import sys
sys.path.append("../../data")

import os
import numpy
from generate import get_data
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def cut(img, region=(55,67)):

    return img[region[0]:region[1], region[0]:region[1]]

def plot_img(dat, ax, vmin=None, vmax=None):

    im = ax.imshow(dat, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.text(0.99, 0.01, "Max: %s" % str(dat.max()),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=10)
    ax.text(0.99, 0.1, "Min: %s" % str(dat.min()),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=10)        

    return im

def plot(fname, odir):

    img_master = get_data(dataset="crtsimg", fname=fname, subimage="master")
    img_master = cut(img_master)

    figmaster, axmaster = plt.subplots(nrows=1, ncols=1)
    im = plot_img(img_master, axmaster)
    figmaster.colorbar(im)
    axmaster.set_title("Master (%s)" % fname)

    plt.savefig(os.path.join(odir, fname + ".master.png"))

    images = []
    for i in xrange(1,5):
        img = get_data(dataset="crtsimg", fname=fname, subimage=i)            
        img = cut(img)
        images.append(img)
    images = numpy.array(images)

    max_image = images.max()
    min_image = images.min()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    for dat, ax in zip(images, axes.flat):
        im = plot_img(dat, ax, vmin=min_image, vmax=max_image)
    
    # Make an axis for the colorbar on the right side
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax)

    plt.savefig(os.path.join(odir, fname + ".sub.png"))

    plt.close()
    plt.close()


if __name__ == "__main__":

    allnames = get_data("crtsimg", imagenames=True)

    odir = "output"

    for i in xrange(len(allnames)):
        fname = allnames[i]
        print("[%i/%i] Generating image %s ..." % (i, len(allnames), fname))
        try:
            plot(fname, odir)
        except Exception as e:
            print("Could not draw image: %s" % str(e))



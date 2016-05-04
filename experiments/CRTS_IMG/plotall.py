import sys
sys.path.append("../../data")

import os
import numpy
from generate import get_data
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from util import plot

if __name__ == "__main__":

    odir = "output"

    # get data via filenames
    X, y, I = get_data("crtsimg", fnames=["1604181570314131003"])

    # plot all images
    for i in xrange(len(y)):
        print("[%i/%i] Generating image %s ..." % (i, len(y), I[i]))
        plot(X[i], y[i], I[i], odir)



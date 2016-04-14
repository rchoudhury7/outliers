import os
import sys
import copy
import shutil
import numpy
import random
import pandas
import urllib

from _generate_util import *

def get_crts_labeled(nrows=None, shuffle=True, seed=0):

    numpy.random.seed(seed)

    filename_big = os.path.join(get_data_path(), "CRTS", "CSDR2_labeled.csv")   
    filename_new = os.path.join(get_data_path(), "CRTS", "allvars.dat")   

    assert check_and_download(filename_big)
    assert check_and_download(filename_new)

    # take updated data (and columns from the previous one)
    df_old = pandas.read_csv(filename_big, sep=",", index_col=0, nrows=nrows)
    df = pandas.read_csv(filename_new, sep=",", index_col=0, nrows=nrows, header=None)
    df.columns = df_old.columns
    
    if shuffle == True:
        df = df.reindex(numpy.random.permutation(df.index))
                
    return df

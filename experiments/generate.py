import os
import sys
import copy
import numpy
import random
import pandas

def get_data_path():

    return os.path.join(os.getcwd().split('outliers')[0], "outliers/data")

def get_CRTS(dataset, nrows=None, shuffle=True, seed=0):

    numpy.random.seed(seed)

    if dataset == "CRTS_Labeled":
        filename_big = os.path.join(get_data_path(), "CRTS", "CSDR2_labeled.csv")   
        filename_new = os.path.join(get_data_path(), "CRTS", "allvars.dat")   
        df = pandas.read_csv(filename_big, sep=",", index_col=0, nrows=nrows)
        df_new = pandas.read_csv(filename_new, sep=",", index_col=0, nrows=nrows, header=None)
        df_new.columns = df.columns
        
        # new dataset
        df = df_new
    
    if shuffle == True:
        df = df.reindex(numpy.random.permutation(df.index))
                
    return df

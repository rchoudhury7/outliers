import os
import sys
import copy
import shutil
import numpy
import random
import pandas
import urllib

def get_data_path():

    return os.path.join(os.getcwd().split('outliers')[0], "outliers/data")

def check_and_download(fname):

    if os.path.isfile(fname) == False:

        try:
        
            urlfname = fname + ".download"
            with open(urlfname,"r") as f:
                url = f.readlines()[0]
                print("Downloading data from %s to %s ..." % (url, fname))
                urllib.urlretrieve (url, fname)
                print("Successfully downloaded the data!")
        except Exception as e:
            print(str(e))
            try:
                # remove incomplete data
                shutil.rmtree(fname)
            except:
                pass
            return False

    return True

def get_CRTS(dataset, nrows=None, shuffle=True, seed=0):

    numpy.random.seed(seed)

    if dataset == "CRTS_Labeled":

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

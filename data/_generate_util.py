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

def update_params(kwargs, default_params):
    """ Updates dictionary with default 
    parameters if needed.
    """

    kwargs = copy.deepcopy(kwargs)
    for key in default_params.keys():
        if key not in kwargs.keys():
            kwargs[key] = default_params[key]

    return kwargs

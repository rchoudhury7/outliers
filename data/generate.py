import os
import sys
import copy
import shutil
import numpy
import random
import pandas
import urllib

from _generate_util import update_params, check_and_download

from _crts30m import get_crts30m
from _crts_labeled import get_crts_labeled
from _crtsimg import get_crtsimages

def get_data(dataset, **kwargs):

    if dataset == "crts_labeled":

        default_params = {'nrows':None, 'shuffle':None, 'seed':0}
        params = update_params(kwargs, default_params)

        return get_crts_labeled(nrows=params['nrows'], \
                                shuffle=params['shuffle'], \
                                seed=params['seed'])

    elif dataset == "crts30m":

        default_params = {'nrows':30000000, 'features':None, 'threshold_n_points':1,'do_select_ra_decl':False,'cache':True,'verbose':1}
        params = update_params(kwargs, default_params)        

        return get_crts30m(nrows=params['nrows'], \
                           features=params['features'], \
                           threshold_n_points=params['threshold_n_points'], \
                           do_select_ra_decl=params['do_select_ra_decl'], \
                           cache=params['cache'], \
                           verbose=params['verbose'])

    elif dataset == "crtsimg":

        default_params = {"fnames":None}
        params = update_params(kwargs, default_params)    


        return get_crtsimages(fnames=params['fnames'])




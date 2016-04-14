import os
import numpy
import time
import pandas
import json

from _generate_util import *

def get_crts30m(nrows=30000000, features=None, threshold_n_points=1, \
             do_select_ra_decl=False, cache=True, verbose=1):
    
    if features is None:
        raise Exception("features must not be None!")

    # search all folders for an appropriate cached version
    if cache == True and os.path.exists(os.path.join(get_data_path(), "CRTS", "cache")):

        subdirs = [x[0] for x in os.walk(os.path.join(get_data_path(), "CRTS", "cache"))]

        for sd in subdirs:
            try:

                with open(os.path.join(sd, 'params.json')) as params_file:    
                    params = json.load(params_file)

                # check if they match with the parameters
                if params['nrows'] == nrows and \
                   params['threshold_n_points'] == threshold_n_points and \
                   params['do_select_ra_decl'] == do_select_ra_decl and \
                   params['features'] == features:

                    # load data
                    data_file = file(os.path.join(sd, "data.npy"), "rb")
                    X = numpy.load(data_file)
                    masterids = numpy.load(data_file)
                    if verbose > 0:
                        print("Using cached version found in %s ..." % str(sd))
                    return X, masterids
            except:
                pass

    # else: parse and store data
    fname = os.path.join(get_data_path(), "CRTS", "crts30m.dat")
    assert check_and_download(fname)

    df = pandas.read_csv(fname, sep="\t", header=0, nrows=2*nrows)
    df = df.dropna()

    # select subset
    df = df[df['Numpoints'] > threshold_n_points]
    if do_select_ra_decl == True:
        quantiles_RASigma = df['RASigma'].quantile(0.95)
        quantiles_DeclSigma = df['DeclSigma'].quantile(0.95)
        a = df['RASigma'] > quantiles_RASigma
        b = df['DeclSigma'] > quantiles_DeclSigma
        selector = ~(a*b)
        df_prep = df[selector]
    else:
        df_prep = df

    df_prep = df_prep[:nrows]

    X = df_prep[features].values
    masterids = df_prep['MasterId']

    if cache==True:

        cache_dir = os.path.join(get_data_path(), "CRTS", "cache", time.strftime("%Y_%m_%d_%H_%M_%S"))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        data_fname = os.path.join(cache_dir, "data.npy")
        params_fname = os.path.join(cache_dir, "params.json")

        params = {'nrows': nrows, 'features':features, 'threshold_n_points':threshold_n_points, 'do_select_ra_decl':do_select_ra_decl}
        with open(params_fname, 'w') as f:
            json.dump(params, f)

        data_file = file(data_fname,"wb")
        numpy.save(data_file, X)
        numpy.save(data_file, masterids)

        data_file.close()

    return X, masterids


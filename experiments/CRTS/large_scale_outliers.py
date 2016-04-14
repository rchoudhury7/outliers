import sys
sys.path.append("..")
sys.path.append("../../data")

import os
import time
import numpy
import pandas
from _large_scale_outliers_util import compute_outliers
from sklearn.preprocessing import StandardScaler
from generate import get_data

ODIR = "large_scale_outliers_output"

# this caches the parsed data; should generate new ones 
# if parameters are changed (e.g., features). Set to 
# 'False' if problems occur ...
cache = True 

# the number of nearest neighbors 
# that are taken into account 
k = 10 
# the number of rows that shall 
# be used for the outlier computation
nrows = 100000
# specifies if a subset shall be considered
do_select_ra_decl = False 
# the number of outlier obects that 
# shall be stored in the file
n_outliers_output = 100 
# the number of 'Numpoints' needed 
# for a pattern to be included
threshold_n_points = 10 

# the features used ...
features = ['Amplitude', 'stetson_j', 'stetson_k', 'Skew', 'fpr_mid35', 'fpr_mid50', 'fpr_mid65', 'fpr_mid80', 'shov', 'maxdiff']
#features = ['Min', 'Max', 'Mean', 'Sd', 'Skew', 'Kurtosis', 'Mad', 'Bwmv', \
#            'thiel_sen', 'durbin_watson', 'stetson_j', 'stetson_k', 'kendall_tau', \
#            'Cusum', 'Con', 'Abbe', 'Reducedchi', 'Amplitude', 'fpr_mid20', \
#            'fpr_mid35', 'fpr_mid50', 'fpr_mid65', 'fpr_mid80', 'shov', 'maxdiff', \
#            'dscore', 'totvar', 'quadvar', 'famp', 'fslop', 'lsd', 'gscore', 'gtvar' \
#           ]

# method
algorithm = "sklearn" #"buffer_kd_tree"
algorithm_params = {'tree_depth':10, \
                    'leaf_size':32, \
                    'n_jobs':8, 'verbose':1, \
                    'n_neighbors': k, \
                    'plat_dev_ids': {0:[3]} \
                }

# get data
X, masterids = get_data("crts30m", nrows=nrows, features=features, threshold_n_points=threshold_n_points, do_select_ra_decl=do_select_ra_decl, cache=cache)

print("Number of remaining objects (after Numpoints threshold selection): %s" % str(len(masterids)))
print("Dimensionality of the feature space (used for selecting the outliers): %s" % str(len(X[0])))

# scale input features
X = StandardScaler().fit_transform(X)

# compute outliers
out_ranks, out_scores, densities = compute_outliers(X, \
                                                    algorithm=algorithm, \
                                                    outlier_criterion="inverse_densities", \
                                                    algorithm_params=algorithm_params)

print("\n\n-------------------------------")
print("------- TOP %i Outliers -------" % n_outliers_output)
print("-------------------------------")
print("MasterId\tScore")
print("-------------------------------")
for i in xrange(n_outliers_output):
    print("%12s" % masterids[out_ranks[i]] + "\t" + "%.8f" % out_scores[out_ranks[i]])
print("-------------------------------")

# write output (top 10000 outliers)
ofilename = os.path.join(ODIR, str(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".csv")
d = os.path.dirname(ofilename)
if not os.path.exists(d):
    os.makedirs(d)
ofile = open(ofilename, 'w')
for i in xrange(len(masterids)):
    line = "%12s" % masterids[out_ranks[i]] + "\t" + "%.8f" % out_scores[out_ranks[i]]
    ofile.write(line + "\n")
ofile.close()


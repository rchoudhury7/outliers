import sys
sys.path.append("../../data")
sys.path.append("../../lib/minisom")

import numpy
from sklearn.preprocessing import StandardScaler
from generate import get_data
from matplotlib import pyplot as plt
from minisom import MiniSom    
from pylab import plot,axis,show,pcolor,colorbar,bone

# this caches the parsed data; should generate new ones 
# if parameters are changed (e.g., features). Set to 
# 'False' if problems occur ...
cache = True 

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

# get data
X, masterids = get_data("crts30m", nrows=nrows, features=features, threshold_n_points=threshold_n_points, do_select_ra_decl=do_select_ra_decl, cache=cache)

# WARNING: take subset
X, masterids = X[:10000], masterids[:10000]

print("Number of remaining objects (after Numpoints threshold selection): %s" % str(len(masterids)))
print("Dimensionality of the feature space (used for selecting the outliers): %s" % str(len(X[0])))

# scale input features
X = StandardScaler().fit_transform(X)

# generate SOM
nx = 10
ny = 10
seed = 0
som = MiniSom(nx, ny, X.shape[1], sigma=0.3, learning_rate=0.5, random_seed=seed, verbose=1)
som.train_random(X, 5*len(X))
#som.train_batch(X, 5*len(X))

counts = numpy.zeros((nx, ny))
for cnt, xx in enumerate(X):
    w = som.winner(xx) # getting the winner
    counts[w] += 1

bone()
# plotting the distance map as background
pcolor(counts, cmap='coolwarm', vmin=0) 
colorbar()
show()

import sys
sys.path.append("../../data")


from sklearn.preprocessing import StandardScaler
from generate import get_data
from matplotlib import pyplot as plt

# this caches the parsed data; should generate new ones 
# if parameters are changed (e.g., features). Set to 
# 'False' if problems occur ...
cache = True 

# the number of rows that shall 
# be used for the outlier computation
nrows = 1000
# specifies if a subset shall be considered
do_select_ra_decl = False 
# the number of outlier obects that 
# shall be stored in the file
n_outliers_output = 100 
# the number of 'Numpoints' needed 
# for a pattern to be included
threshold_n_points = 10 

# the features used ...
#features = ['Amplitude', 'stetson_j', 'stetson_k', 'Skew', 'fpr_mid35', 'fpr_mid50', 'fpr_mid65', 'fpr_mid80', 'shov', 'maxdiff']
features = ['Min', 'Max', 'Mean', 'Sd', 'Skew', 'Kurtosis', 'Mad', 'Bwmv', \
            'thiel_sen', 'durbin_watson', 'stetson_j', 'stetson_k', 'kendall_tau', \
            'Cusum', 'Con', 'Abbe', 'Reducedchi', 'Amplitude', 'fpr_mid20', \
            'fpr_mid35', 'fpr_mid50', 'fpr_mid65', 'fpr_mid80', 'shov', 'maxdiff', \
            'dscore', 'totvar', 'quadvar', 'famp', 'fslop', 'lsd', 'gscore', 'gtvar' \
           ]

# get data
X, masterids = get_data("crts30m", nrows=nrows, features=features, threshold_n_points=threshold_n_points, do_select_ra_decl=do_select_ra_decl, cache=cache)

print("Number of remaining objects (after Numpoints threshold selection): %s" % str(len(masterids)))
print("Dimensionality of the feature space (used for selecting the outliers): %s" % str(len(X[0])))

# scale input features
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

fig, ax = plt.subplots()
ax.grid(True)
ax.scatter(X_r[:,0], X_r[:,1])

for i in xrange(len(X)):
    label = str(i)
    ax.annotate(label, (X_r[i,0], X_r[i,1]))
plt.show()

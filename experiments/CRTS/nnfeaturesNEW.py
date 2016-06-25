import sys
sys.path.append("..")
sys.path.append("../../data")
sys.path.append("../../lib/speedynn")

import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import generate
import exps_util
from speedynn.classification import BruteNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 5
n_selected_features = 2
n_top_combinations = 3

# get dataframe
df = generate.get_data("crts_labeled", shuffle=True)

# get labels and patterns
Y = df['class'].values
features = list(df.columns)
features.remove('class')
X = df[features].values
print("All available classes: %s" % str(set(Y)))

# make binary classification task
X, Y = exps_util.make_binary(X, Y, class1=[2], class2=None)
print("Number of elements for class %i: %i" % (1, (Y==1).sum()))
print("Number of elements for class %i: %i\n" % (2, (Y==2).sum()))

# split up into training and test (and scale the patterns)
train_amount = 0.5
Xtrain, Xtest = X[:int(train_amount*len(X))], X[int(train_amount*len(X)):]
ytrain, ytest = Y[:int(train_amount*len(X))], Y[int(train_amount*len(X)):]
scaler = StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# apply NN model
model = BruteNN(n_selected_features=n_selected_features, n_neighbors=n_neighbors, n_top_combinations=n_top_combinations, verbose=1, n_jobs=2)
model.fit(Xtrain, ytrain)

print("\n\n")
# get predictions and errors for each of the combinations
for i in xrange(len(model.get_top_combinations())):

    # get best combination
    comb = model.get_top_combinations()[i]
    print("Top %i combination (indices): %s" % (i+1, str(comb)))

    # training error
    error = model.get_top_errors()[i]
    print(" -> training error: %f" % (error))

    # get predictions on test set
    preds =  model.predict(Xtest, combination=i)
    # ROHAN: The previous step is the SAME as the following line, have a detailed look at it ;-)
    #preds = KNeighborsClassifier(n_neighbors=n_neighbors).fit(Xtrain[:,comb], ytrain).predict(Xtest[:,comb])
    print(" -> test error: %f" % (1.0 - accuracy_score(ytest,preds)))
    print(" -> MCC on test set: %f\n" % (matthews_corrcoef(ytest, preds)))

# plot result for the BEST combination (index is 0)
print("Plotting best feature combination ...")
top_combination = model.get_top_combinations()[0]
fig, ax = plt.subplots(figsize=(15,15))

# here, we take the subset of the features ("the best combination of two features")
X = X[:,top_combination]
ax.plot(X[:, 0], X[:, 1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=2, alpha=0.5)
ax.plot(X[Y==2, 0], X[Y==2, 1], 'o', markerfacecolor='b', markeredgecolor='k', markersize=2, alpha=0.8)
ax.plot(X[Y==1, 0], X[Y==1, 1], 'o', markerfacecolor='r', markeredgecolor='k', markersize=20, alpha=0.8)
plt.xlabel(features[top_combination[0]], fontsize=12)
plt.ylabel(features[top_combination[1]], fontsize=12)    
plt.title('CRTS')
plt.show()

import sys
sys.path.append("..")
sys.path.append("../../data")
sys.path.append("../../lib/speedynn")

import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import generate
import util
from speedynn.classification import BruteNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef

# get dataframe
df = generate.get_CRTS(dataset="CRTS_Labeled", shuffle=True)

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
model = BruteNN(n_selected_features=2, n_neighbors=5, n_top_combinations=1, verbose=1, n_jobs=4)
model.fit(Xtrain, ytrain)

preds =  model.predict(Xtest)
print("\nClassification error: %f" % (1.0 - accuracy_score(ytest,preds)))
print("MCC: %f" % (matthews_corrcoef(ytest, preds)))

# plot result
top_combination = model.get_top_combinations()[0]
fig, ax = plt.subplots(figsize=(15,15))
X = X[:,top_combination]
ax.plot(X[:, 0], X[:, 1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=2, alpha=0.5)
ax.plot(X[Y==2, 0], X[Y==2, 1], 'o', markerfacecolor='b', markeredgecolor='k', markersize=2, alpha=0.8)
ax.plot(X[Y==1, 0], X[Y==1, 1], 'o', markerfacecolor='r', markeredgecolor='k', markersize=20, alpha=0.8)
plt.xlabel(features[top_combination[0]], fontsize=12)
plt.ylabel(features[top_combination[1]], fontsize=12)    
plt.title('CRTS')
plt.show()

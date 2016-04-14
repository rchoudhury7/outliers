import sys
sys.path.append("..")
sys.path.append("../../data")
sys.path.append("../../lib/speedynn")

import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import generate
import exps_util
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

# get dataframe
df = generate.get_data("crts_labeled", shuffle=True)

# get labels and patterns
Y = df['class'].values
features = list(df.columns)
features.remove('class')
X = df[features].values
print("All available classes: %s" % str(set(Y)))

# make binary classification task
#X, Y = exps_util.make_binary(X, Y, class1=[13], class2=[8,9,10])
X, Y = exps_util.make_binary(X, Y, class1=[2], class2=None)
print("Number of elements for class %i: %i" % (1, (Y==1).sum()))
print("Number of elements for class %i: %i" % (2, (Y==2).sum()))

# split up into training and test (and scale the patterns)
train_amount = 0.5
Xtrain, Xtest = X[:int(train_amount*len(X))], X[int(train_amount*len(X)):]
ytrain, ytest = Y[:int(train_amount*len(X))], Y[int(train_amount*len(X)):]
scaler = StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

model = ExtraTreesClassifier(n_estimators=100, bootstrap=True, max_features=10, criterion="gini", n_jobs=-1, random_state=0)
model.fit(Xtrain, ytrain)

preds =  model.predict(Xtest)
print("\nClassification error: %f" % (1.0 - accuracy_score(ytest,preds)))
print("MCC: %f" % (matthews_corrcoef(ytest, preds)))




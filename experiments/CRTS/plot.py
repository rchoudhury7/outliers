import sys
sys.path.append("..")

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import generate
import util

selected_features = ['beyond1std', 'flux_percentile_ratio_mid35'] #['ls', 'amplitude']

# get data
df = generate.get_CRTS(dataset="CRTS_Labeled", shuffle=True)

print df.columns.values
Y = util.make_one_vs_all(df['class'].values, target_class=8, label_class_target=1, label_class_others=2)
X = df[selected_features].values

# plot data
fig, ax = plt.subplots(figsize=(15,15))

ax.plot(X[:, 0], X[:, 1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=2, alpha=0.5)
print("Number of elements in class 1: %i" % len(X[Y==1, 0]))
print("Number of elements in class 2: %i" % len(X[Y==2, 0]))
ax.plot(X[Y==1, 0], X[Y==1, 1], 'o', markerfacecolor='r', markeredgecolor='k', markersize=10, alpha=0.8)
ax.plot(X[Y==2, 0], X[Y==2, 1], 'o', markerfacecolor='b', markeredgecolor='k', markersize=10, alpha=0.8)
plt.xlabel(selected_features[0], fontsize=12)
plt.ylabel(selected_features[1], fontsize=12)    
plt.title('CRTS')
plt.show()

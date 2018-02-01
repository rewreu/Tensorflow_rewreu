import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

X, y = batch_data, batch_labels.reshape(-1)
# X, y = df, batch_labels.reshape(-1)
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=150,
                              random_state=135)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(16,10))
plt.title("Feature importances",fontsize=40)
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices, fontsize = 30)


plt.xlim([-1, X.shape[1]])
plt.show()

from sklearn.svm import SVC

svc = SVC().fit(X_train, y_train)
svc.
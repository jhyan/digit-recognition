import numpy as np
from numpy.matlib import repmat
import sys
import time
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, datasets, svm


# load data
xTr = np.loadtxt("trainX.csv", delimiter=",")
yTr = np.loadtxt("trainY.csv", delimiter=",")
xTe = np.loadtxt("testX.csv", delimiter=",")
print ("xTr.shape: ",xTr.shape)
print ("yTr.shape: ",yTr.shape)
print ("xTe.shape: ",xTe.shape)
k_fold = KFold(n_splits=5)

############# random forest
# clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, \
#                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
#                               max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, \
#                               n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
# clf.fit(xTr, yTr)
# res = clf.predict(xTe)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# np.savetxt("prediction.csv", res, fmt="%s", delimiter=",")
# # [ 0.77434846  0.79888089  0.79025309  0.8270994   0.75454238]
# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


############# knn. 1nn the best, 0.93
# for i in range(1,2):
#     clf = neighbors.KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', \
#                                           leaf_size=30, p=2, metric='minkowski', metric_params=None, \
#                                           n_jobs=1, **kwargs)
#     clf.fit(xTr, yTr)
#     res = clf.predict(xTe)
#     print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
#     np.savetxt("prediction_knn.csv", res, fmt="%s", delimiter=",")
#     #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


############## svm
clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,\
                decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',\
                max_iter=-1, probability=False, random_state=None, shrinking=True,\
                tol=0.001, verbose=False)
clf.fit(xTr, yTr)
res = clf.predict(xTe)
print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# np.savetxt("prediction_svm.csv", res, fmt="%s", delimiter=",")
# [ 0.65516836  0.67049135  0.65312592  0.688606    0.63911956]
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html








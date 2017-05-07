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

# random forest
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(xTr, yTr)
# res = rf.predict(xTe)
# print ("results: ", type(res))
# np.savetxt("prediction.csv", res, fmt="%s", delimiter=",")

# knn
for i in range(1,10):
    clf = neighbors.KNeighborsClassifier(i)
    clf.fit(xTr, yTr)
    res = clf.predict(xTe)
    print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# np.savetxt("prediction_knn.csv", res, fmt="%s", delimiter=",")

# # svm
# clf = svm.SVC()
# clf.fit(xTr, yTr)
# res = clf.predict(xTe)
# # np.savetxt("prediction_svm.csv", res, fmt="%s", delimiter=",")









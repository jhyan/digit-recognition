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
from sklearn.neural_network import MLPClassifier

# load data
xTr = np.loadtxt("trainX.csv", delimiter=",")
yTr = np.loadtxt("trainY.csv", delimiter=",")
xTe = np.loadtxt("testX.csv", delimiter=",")
print ("xTr.shape: ",xTr.shape)
print ("yTr.shape: ",yTr.shape)
print ("xTe.shape: ",xTe.shape)
k_fold = KFold(n_splits=4)

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


################## svm
gamma_range = np.logspace(-3, 3, 3)
c_range = np.logspace(0,3,3)
for c in c_range:
    for g in gamma_range:
        clf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,\
                        decision_function_shape='ovo', degree=3, gamma = g, kernel='poly',\
                        max_iter=-1, probability=False, random_state=None, shrinking=True,\
                        tol=0.001, verbose=False) # gamma = "auto"
        clf.fit(xTr, yTr)
        res = clf.predict(xTe)
        print (c, g)
        print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# np.savetxt("prediction_svm.csv", res, fmt="%s", delimiter=",")
# [ 0.8227374   0.83417365  0.81352299  0.82787335  0.80122374] for C=1000. larger C better here
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


# ################### neuron network
# clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', \
#                     alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,\
#                     power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, \
#                     warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, \
#                     validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# clf.fit(xTr, yTr)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# res = clf.predict(xTe)
# # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier


# ############## Stochastic Gradient Descent ############ default hinge so linear svm
# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier()
# clf.fit(xTr, yTr)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))


# # other useful links
# https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/s
# http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2

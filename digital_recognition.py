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
from scipy.ndimage import rotate

# load data
xTr = np.loadtxt("trainX.csv", delimiter=",")
n,d = xTr.shape
yTr = np.loadtxt("trainY.csv", delimiter=",")
xTe = np.loadtxt("testX.csv", delimiter=",")
k_fold = KFold(n_splits=4)

######### extend the sample by random data
# mu, sigma = 0, 0.2 # mean and standard deviation
# ran = np.random.normal(mu, sigma, n*d).reshape(n,d)
# ran = np.random.rand(xTr.shape[0],xTr.shape[1])/5 - 0.1
# print (ran[0])

# xTr_extra = xTr + ran
# print (xTr_extra[0])
# idx = np.arange(len(xTr) + len(xTr_extra))
# xTr =  np.concatenate((xTr, xTr_extra),axis=0)[idx]
# yTr =  np.concatenate((yTr, yTr_extra),axis=0)[idx]



######### rotate and get 97.5%
angle_list = [-27, -20, -13, -6, 6, 13, 20, 27]
for angle in angle_list:
    xTr_extra = np.zeros((n, d))
    for i in range(n):
        xTr_extra[i] = rotate(xTr[i].reshape(28,28), angle, reshape=False).reshape(1,d) + np.random.rand(1,d)/5 - 0.1 # rotate and add random noise
    xTr = np.concatenate((xTr, xTr_extra), axis=0) # add (n,d) data each time
    yTr = np.concatenate((yTr, yTr[:n:]),axis=0) # add (n,1) yTr

idx = np.arange(len(xTr))
np.random.shuffle(idx)


print ("xTr.shape: ",xTr.shape)
print ("yTr.shape: ",yTr.shape)
print ("xTe.shape: ",xTe.shape)

# ############## random forest
# clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, \
#                               min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_features='auto', \
#                               max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, \
#                               n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
# clf.fit(xTr, yTr)
# res = clf.predict(xTe)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# np.savetxt("prediction_rf.csv", res, fmt="%s", delimiter=",")
# # [ 0.77434846  0.79888089  0.79025309  0.8270994   0.75454238]
# # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


# ############ knn. 1nn the best
# for i in range(1,2):
#     clf = neighbors.KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', \
#                                           leaf_size=30, p=2, metric='minkowski', metric_params=None, \
#                                           n_jobs=-1)
#     clf.fit(xTr, yTr)
#     res = clf.predict(xTe)
#     print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
#     np.savetxt("prediction_knn.csv", res, fmt="%s", delimiter=",")
#     #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# # [ 0.99790692  0.99946304  1. 0.98346916] with 8 rotations and random noise

################## svm
gamma_range = np.logspace(-2, 2, 3)
c_range = np.logspace(0, 3, 3)
# for c in c_range:
    # for g in gamma_range:
clf = svm.SVC(C= 10, cache_size=200, class_weight=None, coef0=0.0,\
                decision_function_shape='ovo', degree=3, gamma = 1, kernel='poly',\
                max_iter=-1, probability=False, random_state=None, shrinking=True,\
                tol=0.001, verbose=False) # gamma = "auto"
clf.fit(xTr, yTr)#
res = clf.predict(xTe)
# print (c, g)
print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
np.savetxt("prediction_svm.csv", res, fmt="%s", delimiter=",")
# [ 0.98510352  0.99806012  0.99944894  0.94175308] for C=10 larger C better here
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


# ################### neuron network
# clf = MLPClassifier(hidden_layer_sizes=(500), activation='relu', solver='adam', \
#                     alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,\
#                     power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, \
#                     warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, \
#                     validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# clf.fit(xTr, yTr)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))
# res = clf.predict(xTe)
# np.savetxt("prediction_NN.csv", res, fmt="%s", delimiter=",")
# # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier


# ############## Stochastic Gradient Descent ############ default hinge so linear svm
# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier()
# clf.fit(xTr, yTr)
# print (cross_val_score(clf, xTr, yTr, cv=k_fold, scoring='precision_macro'))


# # other useful links
# https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/s
# http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2

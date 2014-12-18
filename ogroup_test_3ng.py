

import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier
import baseline
import ogroup


import random
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

import get_3ng_aug_data as ng

Xnew, y_train, groups, dict_text = ng.get_data()

X_train = Xnew
y_train = y_train


print "### BASELINE GROUP LASSO in pure python/numpy###"
X = X_train
y = y_train
clf = ogroup.BaselineGroupLasso(max_iter=30, alpha=.01,  max_steps=30)
clf.fit(X, y, groups)
print "Acc:", clf.score(X, y)
print (clf.coefs_)


print "### Equivalent Lightning Cython Implementation ###"
light_clf = CDClassifier(penalty="l1/l2",
                         loss="squared_hinge",
                         multiclass=True,
                         max_iter=clf.max_iter,
                         alpha=1e-4, # clf.alpha,
                         C=1.0 / X.shape[0],
                         tol=clf.tol,
                         permute=False,
                         verbose=3,
                         random_state=0).fit(X, y)
print "Acc:", light_clf.score(X, y)
print (light_clf.coef_.T)


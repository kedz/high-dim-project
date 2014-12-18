

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

np.set_printoptions(precision=4)

Xnew, y_train, groups, dict_text = ng.get_data()

X_train = Xnew
y_train = y_train


print "### BASELINE GROUP LASSO in pure python/numpy###"
X = X_train
y = y_train
clf = ogroup.BaselineGroupLasso(max_iter=50, alpha=.01,  max_steps=50)
clf.fit(X, y, groups)
print "Acc:", clf.score(X, y)
print (clf.coefs_)


import heapq

top_words =  30

for m in xrange(clf.coefs_.shape[1]):
	t = []
	print 'Topic',m
	for row in xrange(clf.coefs_.shape[0]):
		if( clf.coefs_[row,m] >0):
			t.append( dict_text[row])
	for k in heapq.nlargest(top_words,t):
		print k,
	print
 

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



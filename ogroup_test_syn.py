

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

m_classes = 10
m_features = 100
m_groups = 10
m_no_in_each_group = m_features/m_groups
multiple = 1.3
m_samples = 1000
attemp = 0

while 1:
    attemp +=1
    print 'attemp', attemp
    groups = []
    weights = np.zeros((m_features,m_classes))
    idx = 0
    for g in xrange(m_groups):
        groups.append( (g*m_no_in_each_group, (g+1)*m_no_in_each_group) )
        #seed = idx % m_classes
        for m in xrange(m_classes):
            #if seed != m :
            #    continue
            seed = random.random()
            if ( seed < multiple/m_classes):
                #print 'chosen column', m
                for i in xrange(g*m_no_in_each_group, (g+1)*m_no_in_each_group):
                    #weights[i,m] = (round(random.random(),3) )
                    weights[i,m] = (round(random.random(),3) - 0.5 )*2
        idx += 1
    X = np.random.rand(m_samples,m_features)
    pred = safe_sparse_dot(X, weights)
    y_pred = np.argmax(pred, axis=1)
    print 'unique classes',  np.unique(y_pred).shape[0]
    if np.unique(y_pred).shape[0] == m_classes:
        break

print 'weights', weights
print 'groups', groups
print 'y_pred', y_pred
print '======================================================'

X_train = X
y_train = y_pred


print "### BASELINE GROUP LASSO in pure python/numpy###"
X = X_train
y = y_train
clf = ogroup.BaselineGroupLasso(max_iter=30, alpha=.5, max_steps=30)
clf.fit(X, y, groups)
print "Acc:", clf.score(X, y)
print clf.coefs_


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
print light_clf.coef_.T


exit(0)

import numpy as np
data = np.load('3ng_train.npz')
X = data['X'].item()
Xaug = data['Xaug'].item()
y = data['y']
groups = data['groups']
clf.fit(Xaug, y, groups)
print clf.score(Xaug, y)

light_clf.verbose=1
light_clf.fit(X, y)

print light_clf.score(X, y)

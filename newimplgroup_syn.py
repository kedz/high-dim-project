import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier
import random
import sys

class ColumnData(object):
    def __init__(self, X):
        self._X = scipy.sparse.csc_matrix(X)
    def get_row(self,row):
        return self._X[row,:]
    def get_column(self, col):
        """Iterator over (row, val) tuples for nonzero values in column col"""
        for ii in range(self._X.indptr[col], self._X.indptr[col + 1]):
           yield (self._X.indices[ii],  self._X.data[ii])

def calculate_AW(ds, y, n_samples, n_classes,coefs_):
    AW =  np.ones((n_samples, n_classes))
    for i in xrange(n_samples):
        for r in xrange(n_classes):
            Xi=ds.get_row(i)
            AW[i,r]-=safe_sparse_dot(Xi,coefs_[:,y[i]]-coefs_[:,r])
    return AW

def calculate_loss(ds, y, n_samples, n_features, n_classes,coefs_, lamb, groups):
    AW = calculate_AW(ds,y, n_samples, n_classes,coefs_)
    LOSS = 0
    LAMBDA = lamb
    for i in xrange(n_samples):
        for r in xrange(n_classes):
            if y[i] != r:
                #print i,y[i],r
                LOSS += max(AW[i,r],0) ** 2    
    LOSS/=float(n_samples)
    for group in groups:
        for m in xrange(n_classes):
            LOSS += np.linalg.norm(coefs_[group[0]:group[1],m], 2) * LAMBDA
    return LOSS
 
def _derivatives(n_classes, j, ds, y, AW, one_over_n, m):
    Gj = np.zeros((n_classes))
    hj = np.zeros((n_classes))
    for r in xrange(n_classes):
        for i, Xij in ds.get_column(j):
            if y[i] != r and AW[i,r] > 0:
                if y[i] == m:
                    Gj[y[i]] -= AW[i,r] * Xij
                    hj[y[i]] += Xij * Xij
                if r == m:
                    Gj[r] += AW[i,r] * Xij
                    hj[r] += Xij * Xij
    Gj[m] *= 2 * one_over_n
    hj[m] *= 2 * one_over_n
    return Gj, hj

def fit(ds, y, one_over_n, n_samples, n_features, n_classes,coefs_,groups):
    lamb = 0.5
    tol = 1e-3
    prevl = -1
    for k in xrange(15):
        loss = calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb,groups)
        if abs(loss-prevl)<tol:
            break
        prevl=loss
        for group in groups:
            Gblock=np.ones(group[1]-group[0])
            for m in xrange(n_classes):
                AW = calculate_AW(ds, y, n_samples, n_classes,coefs_)
                Lblock=1e-4
                for j in xrange(group[0],group[1]):
                    Gj, hj = _derivatives(n_classes, j, ds, y, AW, one_over_n, m)
                    Gblock[j-group[0]]=Gj[m]
                    Lblock=max(Lblock,hj[m])
                Lblock = min(Lblock, 1e9)
                Vblock=coefs_[group[0]:group[1],m] - Gblock/Lblock
                #print Vblock
                muj = lamb/Lblock
                #print muj
                L2 = np.linalg.norm(Vblock,2)
                if L2 !=0 :
                    Wblock=max(1- muj/ np.linalg.norm(Vblock,2),0)*Vblock
                else:
                    Wblock=np.zeros(group[1]-group[0])
                loss = calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb,groups)
                Wblock_old = coefs_[group[0]:group[1],m].copy()
                delta = Wblock - Wblock_old
                #print "delta", delta
                max_loop = 10
                alpha = 0.5
                alphas = 1
                while max_loop > 0 :
                    #print "coefs",coefs_,"loss",loss
                    coefs_[group[0]:group[1],m] = Wblock_old + alphas*delta
                    newLoss = calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb, groups)
                    #print "new coefs", coefs_,"new loss", newLoss
                    if newLoss < loss:
                        print "loss: ", loss
                        #print "better",group[0],m
                        break
                    alphas *= alpha
                    max_loop -= 1
                if max_loop ==0:
                    coefs_[group[0]:group[1],m] = Wblock_old
    print coefs_

def score(X, y, coefs_):
    pred = safe_sparse_dot(X, coefs_)
    y_pred = np.argmax(pred, axis=1)
    print "pred =", pred
    print "y_pred =", y_pred
    print "y =", y
    n_correct = np.where(y_pred == y)[0].shape[0]
    return n_correct / float(X.shape[0])

np.set_printoptions(precision=3)

m_classes = 5
m_features = 25
m_groups = 5
m_no_in_each_group = m_features/m_groups
multiple = 1.3
m_samples = 150
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
print 'press enter to start regression'
sys.stdin.readline()

X_train = X
y_train = y_pred


X = X_train
y = y_train
n_features = X.shape[1]
n_classes = np.unique(y).shape[0]
n_samples = X.shape[0]
one_over_n = 1. / float(n_samples)
ds = ColumnData(X)
coefs_ = np.zeros((n_features, n_classes))

fit( ds, y, one_over_n, n_samples, n_features, n_classes,coefs_,groups)
s =  score (X,y,coefs_)
print "score = ", s

clf_max_iter=300
clf_tol = 1e-3
print "### Equivalent Lightning Cython Implementation ###"
light_clf = CDClassifier(penalty="l1/l2",
                         loss="squared_hinge",
                         multiclass=True,
                         max_iter=clf_max_iter,
                         alpha=0.5, # clf.alpha,
                         C=1.0 / X.shape[0],
                         tol=clf_tol,
                         permute=False,
                         verbose=3,
                         random_state=0).fit(X, y)
print "Acc:", light_clf.score(X, y)
print light_clf.coef_.T






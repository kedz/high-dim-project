import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier

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

def calculate_loss(ds, y, n_samples, n_features, n_classes,coefs_, lamb):
    AW = calculate_AW(ds,y, n_samples, n_classes,coefs_)
    LOSS = 0
    LAMBDA = lamb
    for i in xrange(n_samples):
        for r in xrange(n_classes):
            if y[i] != r:
                #print i,y[i],r
                LOSS += max(AW[i,r],0) * max(AW[i,r],0)    
    LOSS/=float(n_samples)
    for r in xrange(n_features):
        LOSS += np.linalg.norm(coefs_[r,:], 2) * LAMBDA
    return LOSS
 
def _derivatives(n_classes, j, ds, y, AW, one_over_n):
    Gj = np.zeros((n_classes))
    hj = np.zeros((n_classes))
    for r in xrange(n_classes):
        for i, Xij in ds.get_column(j):
            if y[i] != r and AW[i,r] > 0:
                Gj[y[i]] -= AW[i,r] * Xij
                Gj[r] += AW[i,r] * Xij
                hj[y[i]] += Xij * Xij
                hj[r] += Xij * Xij
    Lpp_max = -1<<31
    for k in xrange(n_classes):
        Gj[k] *= 2 * one_over_n
        Lpp_max = max(Lpp_max, hj[k])
    Lpp_max *= 2 * one_over_n
    print "raw Lpp_max", Lpp_max, "Gj", Gj
    Lpp_max = min(max(Lpp_max, 1.0e-4), 1e9)
    return Gj, Lpp_max

def fit(ds, y, one_over_n, n_samples, n_features, n_classes,coefs_):
    lamb = 0.05
    tol = 1e-5
    prevl=-1
    for k in xrange(15):
        loss = calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb)
        if abs(loss-prevl)<tol:
            break
        prevl=loss
        for j in xrange(0,n_features):
            AW = calculate_AW(ds, y, n_samples, n_classes,coefs_)
            Gj, Lj = _derivatives( n_classes, j, ds, y, AW, one_over_n)
            Vj = coefs_[j,:] - Gj / Lj
            print "Vj", Vj
            muj = lamb/Lj
            L2 = np.linalg.norm(Vj,2)
            if L2 !=0 :
                Wj=max(1- muj/ np.linalg.norm(Vj,2),0)*Vj
            else:
                Wj=np.zeros(n_classes)
            print "Wj", Wj
            loss = calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb)
            print "loss: ", loss
            Wj_old = coefs_[j,:].copy()
            print "Wj_old", Wj_old
            delta = Wj - Wj_old
            print "delta", delta
            max_loop = 100
            alpha = 0.5
            alphas = 1
            while max_loop > 0 :
                #print "coefs",coefs_,"loss",loss
                coefs_[j,:] = Wj_old + alphas*delta
                newLoss =  calculate_loss(ds,y, n_samples,n_features, n_classes,coefs_,lamb)
                #print "new coefs", coefs_,"new loss", newLoss
                if newLoss < loss:
                    print "better"
                    break
                alphas *= alpha
                max_loop -= 1
            if max_loop ==0:
                coefs_[j,:] = Wj_old
            print coefs_

def score( X, y, coefs_):
    pred = safe_sparse_dot(X, coefs_)
    y_pred = np.argmax(pred, axis=1)
    print "pred =", pred
    print "y_pred =", y_pred
    print "y =", y
    n_correct = np.where(y_pred == y)[0].shape[0]
    return n_correct / float(X.shape[0])

documents = [
"Graph Human machine interface for lab abc computer applications",
"A survey of user opinion of computer system response time",
"The EPS user interface management system",
"System and human system engineering testing of EPS",
"Relation of user perceived response time to error measurement",

"The generation of random binary unordered trees",
"The intersection graph of paths in trees",

"Graph minors IV Widths of trees and well quasi ordering",
"Graph minors A survey"]

labels = [0,0,0,0,0,1,1,2,2]
#labels=[0,1,2]
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
for document in documents]
# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
for text in texts]
#print texts
#exit(0)

texts = [ " ".join(text) for text in texts]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
y_train = labels

a = []
for k in vectorizer.vocabulary_:
    a.append(  "%03d" % (vectorizer.vocabulary_[k],)+" "+k) 

a.sort()
for i in a:
    print i    

X = X_train
y = y_train
n_features = X.shape[1]
n_classes = np.unique(y).shape[0]
n_samples = X.shape[0]
one_over_n = 1. / float(n_samples)
ds = ColumnData(X)
coefs_ = np.zeros((n_features, n_classes))

fit( ds, y, one_over_n, n_samples, n_features, n_classes,coefs_)
score (X,y,coefs_)

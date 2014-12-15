import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier
import baseline
import ogroup

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
assert(len(a)==12)

groups=((0,3),(3,6),(6,10),(10,12))
#groups=[]
#for x in xrange(12):
#    groups.append((x,x+1))
print groups



print "### BASELINE GROUP LASSO in pure python/numpy###"
X = X_train
y = y_train
clf = ogroup.BaselineGroupLasso(max_iter=30, alpha=.001, max_steps=30)
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

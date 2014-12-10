import numpy as np
import mmclf
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier

# newsgroups_train = fetch_20newsgroups(
#     subset='train', remove=('headers', 'footers', 'quotes'))

# newsgroups_test = fetch_20newsgroups(
#     subset='test', remove=('headers', 'footers', 'quotes'))

# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(newsgroups_train.data)
# y_train = newsgroups_train.target

# data=np.load("3ng_train.npz")
# X_train=data["X"].item()
# y_train=data["y"]
# print X_train, y_train
# groups=data["groups"]

# print "X train dims:", X_train.shape

# groups=[]
# for i in range(0, X_train.shape[1]-2000, 2000):
#     groups.append((i,i+2000))
# groups.append((i+2000,X_train.shape[1]))

# print "fake groups:", groups

# X_test = vectorizer.transform(newsgroups_test.data)
# y_test = newsgroups_test.target
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

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
for text in texts]

texts = [ " ".join(text) for text in texts]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
y_train = labels

clf = CDClassifier(penalty="l1/l2",
                   loss="squared_hinge",
                   multiclass=True,
                   max_iter=15,
                   alpha=1e-4,
                   C=1.0 / X_train.shape[0],
                   tol=1e-6, verbose=5)


mmclf = mmclf.LatentGroupClassifier(max_iter=15, C=1.0 / X_train.shape[0])
start = time()
clf.fit(X_train, y_train)
elapsed = time() - start
print "CDClassifier time", elapsed
print "CDClassifier score", clf.score(X_train, y_train)
start = time()
mmclf.fit(X_train, y_train)
elapsed = time() - start
print "LatentGroupClassifier time", elapsed
print "LatentGroupClassifier score", mmclf.score(X_train, y_train)
print "CDClassifier weights\n", clf.coef_.T
print "LatentGroupClassifier weights\n", mmclf.coefs_.T
print "features", vectorizer.vocabulary_

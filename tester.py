import numpy as np
import mmclf
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from lightning.classification import CDClassifier

newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'))

newsgroups_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target


X_test = vectorizer.transform(newsgroups_test.data)
y_test = newsgroups_test.target


#clf = CDClassifier(penalty="l1/l2",
#                   loss="squared_hinge",
#                   multiclass=True,
#                   max_iter=5,
#                   alpha=1e-4,
#                   C=1.0 / X_train.shape[0],
#                   tol=1e-6, verbose=5)
#

clf = mmclf.LatentGroupClassifier(max_iter=5, C=.001)
start = time()
clf.fit(X_train, y_train)
elapsed = time() - start
print elapsed
print clf.score(X_test, y_test)

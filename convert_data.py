import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

newsgroups_train = fetch_20newsgroups(
    categories=['alt.atheism','comp.graphics'],
    subset='train', remove=('headers', 'footers', 'quotes'))

#newsgroups_test = fetch_20newsgroups(
#    subset='test', remove=('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target

X_train_bin = X_train.copy()
X_train_bin.data[:] = 1

XX = X_train_bin.T.dot(X_train_bin)
XX.data[:] = 1

print "X train dims:", X_train.shape
print "XX nonzeros:", len(XX.data)


index_sets = set()
for i in range(XX.shape[0]):
    row = XX.getrow(i)
    index_set = list(ii for ii in row.indices)
    index_set.sort()
    index_set = tuple(index_set)
    if len(index_set) > 0:
        index_sets.add(index_set)
    print i, XX.shape[0]

index_sets = list(index_sets)
n_dims = 0
for index_set in index_sets:
    n_dims += len(index_set)

print "Num of index sets:", len(index_sets)
print "Total dimensions:", n_dims

#X_train = X_train.tocsc()

Xnew = scipy.sparse.lil_matrix((X_train.shape[0], n_dims))
xnew_ii = 0
orig_indices = []
groups = []
for index_set in index_sets:
    group_start = xnew_ii
    for index in index_set:
        col = X_train.getcol(index)
        for r in col.indices:
            Xnew[r,xnew_ii] = X_train[r, index]
        xnew_ii += 1
        orig_indices.append(index)
    group_end = xnew_ii
    groups.append([group_start, group_end])


groups = np.array(groups)
orig_indices = np.array(orig_indices)
np.savez_compressed(
    "20ng_train.npz", groups=groups, orig_indices=orig_indices,
    Xaug=Xnew, y=y_train, X=X_train)

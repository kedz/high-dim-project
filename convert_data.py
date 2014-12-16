import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import scipy
from sklearn.neighbors import KDTree, NearestNeighbors
from nltk import PorterStemmer

stem = PorterStemmer().stem_word

newsgroups_train = fetch_20newsgroups(
    categories=['alt.atheism', 'comp.graphics', 'misc.forsale'],
    subset='train', remove=('headers', 'footers', 'quotes'))

f = open("stopwords")
lines = f.readlines()
stoplist = set([ line.strip() for line in lines])

##### get word frequency ####
word_pattern = re.compile(r"^[a-z']+$")

texts = [[stem(word) for word in document.lower().split() 
			if word_pattern.match(word)!= None and
			word not in stoplist and
			len(word)>1 ]
			for document in newsgroups_train.data]
			
word_counts ={}
for text in texts:
	for w in text:
		word_counts[w] = word_counts.get(w,0) + 1

min_count = 5
infrequent_words = set()

for word, count in word_counts.iteritems():
	if(count < min_count):
		infrequent_words.add(word)

##### filter out infrequent words ####
texts = [{word: word_counts[word] for word in text 
			if word not in infrequent_words}
			for text in texts]

print len(texts)
#X_train = 
vec = DictVectorizer(dtype=np.int32)
X_train = vec.fit_transform(texts)

trans = TfidfTransformer()
X_train = trans.fit_transform(X_train)


print X_train.shape

#newsgroups_test = fetch_20newsgroups(
#    subset='test', remove=('headers', 'footers', 'quotes'))

#vectorizer = TfidfVectorizer()
#X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target

#X_train_bin = X_train.copy()
#X_train_bin.data[:] = 1

#XX = X_train_bin.T.dot(X_train_bin)
#XX.data[:] = 1

print "X train dims:", X_train.shape
#print "XX nonzeros:", len(XX.data)
#print XX

from sklearn.cluster import KMeans

#XX = XX.todense()
#nn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(XX)
#distances, neighbors = nn.kneighbors(XX)
#used = set()
#for dist, nbors in zip(distance, neighbors):
#    if nbors[0] in used:
#        continue
n_cluster = 500
km = KMeans(
    n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300,
    tol=0.0001, precompute_distances=True, verbose=1, random_state=None,
    copy_x=True, n_jobs=24) 
km.fit(X_train)

n_classes = np.unique(y_train).shape[0]
n_dims = 0
index_sets = list()


for i in range(n_cluster):
    index_set = set()
    label_counts = np.zeros((n_classes))
    for x in np.where(km.labels_ == i)[0]:
        label_counts[y_train[x]] += 1
        #print "\tx =", x
        for feat in X_train.getrow(x).indices:
            index_set.add(feat)

#    index_set = [idx for idx in np.where(km.labels_ == i)[0]]
    print i, 
    label_counts / np.sum(label_counts).astype(np.float64)
    index_set = list(index_set)
    if len(index_set) > 0:
        purity = label_counts / np.sum(label_counts).astype(np.float64)
        index_sets.append((purity, index_set))

index_sets.sort(key=lambda x: np.max(x[0]), reverse=True)

topn = 200
for purity, indices in index_sets[:topn]:
    print purity         

index_sets = [tuple(sorted(list(index_set)))
              for p, index_set in index_sets[:topn]]

n_dims = 0
for index_set in index_sets:
    n_dims += len(index_set)

print "Num of index sets:", len(index_sets)
print "Total dimensions:", n_dims


#for i in range(n_cluster):
#    #print "cluster", i
#    index_set = set()
#    label_counts = np.zeros((n_classes))
#    for x in np.where(km.labels_ == i)[0]:
#        label_counts[y_train[x]] += 1
#        #print "\tx =", x
#        for feat in X_train.getrow(x).indices:
#            index_set.add(feat)
##    index_set = [idx for idx in np.where(km.labels_ == i)[0]]
#    print i, 
#    label_counts / np.sum(label_counts).astype(np.float64)
#    index_set = list(index_set)
#    index_set.sort()
#    #print index_set
#
#    if len(index_set) > 0:
#        n_dims += len(index_set)       # print "\t", idx
#        index_sets.add(tuple(index_set))
#



#kdt = KDTree(XX, #leaf_size=30,
#             metric='euclidean')
#print kdt.query(XX, k=2, return_distance=False) 



#import sys
#sys.exit()
#
#index_sets = set()
#for i in range(XX.shape[0]):
#    row = XX.getrow(i)
#    index_set = list(ii for ii in row.indices)
#    index_set.sort()
#    index_set = tuple(index_set)
#    if len(index_set) > 0:
#        index_sets.add(index_set)
#    print i, XX.shape[0]
#
#index_sets = list(index_sets)
#n_dims = 0
#for index_set in index_sets:
#    n_dims += len(index_set)

#print "Num of index sets:", len(index_sets)
#print "Total dimensions:", n_dims

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
print groups
Xnew = Xnew.tocsr()
np.savez_compressed(
    "3ng_train.npz", groups=groups, orig_indices=orig_indices,
    Xaug=Xnew, y=y_train, X=X_train)
with open("vocab.txt", "w") as f:
    for fname in vec.feature_names_:
        f.write("{}\n".format(fname))
        #print fname


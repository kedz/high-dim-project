

import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import scipy
from sklearn.neighbors import KDTree, NearestNeighbors
from nltk import PorterStemmer

def get_data():
	stem = PorterStemmer().stem_word

	categories=['alt.atheism', 'comp.graphics', 'misc.forsale']
	print "Retrieving articles from categories :",categories

	newsgroups_train = fetch_20newsgroups(
		categories=categories,
		subset='train', remove=('headers', 'footers', 'quotes'))

	newsgroups_train.data = newsgroups_train.data[0:200]
	newsgroups_train.target = newsgroups_train.target[0:200]

	print "Pre-processing ... "
	f = open("stopwords")
	lines = f.readlines()
	stoplist = set([ line.strip() for line in lines])

	##### get word frequency ####
	word_pattern = re.compile(r"^[a-z]+$")

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

	print "Number of articles : ",len(texts)

	vec = DictVectorizer(dtype=np.int32)
	X_train = vec.fit_transform(texts)

	trans = TfidfTransformer()
	X_train = trans.fit_transform(X_train)

	y_train = newsgroups_train.target

	X_train_bin = X_train.copy()
	##############################

	import heapq


	def sim_words(XX, row_i,already_chosen):
		row = row_i
		words = []
		t = []
		top_size = 10
		for i in xrange(XX.shape[0]):
			if ( i == row_i):
				continue
			v = XX[row,i]
			if(v >0):
				t.append((v,i))
		words.append(row)
		for (k, v) in heapq.nlargest(top_size,t):
			words.append(v)
			for j in xrange(row_i, XX.shape[0]):
				if(XX[j,v] ==0):
					continue
				XX[j,v] = 0
			already_chosen.add(v)
		return words

	print "Calculating co-occurence matrix and transforming data ..."
	XX = X_train_bin.T.dot(X_train_bin)
	already_chosen = set()
	groups = []
	dict = []
	dict_text = []

	for i in xrange(XX.shape[0]):
		if( i in already_chosen):
			continue
		words = sim_words(XX,i,already_chosen)
		#groups.append(words)
		dict_len1 = len(dict)
		for w in words:
			dict_text.append(vec.feature_names_[w])
			dict.append(w)
		dict_len2 = len(dict)
		groups.append((dict_len1, dict_len2))


	n_newcols = len(dict)
	Xnew = scipy.sparse.lil_matrix((X_train.shape[0], n_newcols))

	n_rows =X_train.shape[0] 
	n_cols =X_train.shape[1] 

	for row in xrange(n_rows):
		for col in xrange(n_newcols):
			Xnew[row, col] = X_train[row,dict[col]]
	
	return Xnew, y_train, groups, dict_text



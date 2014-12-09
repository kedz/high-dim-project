#install gensim: http://radimrehurek.com/gensim/install.html

import re
from gensim import corpora, models, similarities
from sklearn.datasets import fetch_20newsgroups

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'))

f = open("stopwords")
lines = f.readlines()
stoplist = set([ line.strip() for line in lines])

##### get word frequency ####
word_pattern = re.compile(r"^[a-z']+$")

texts = [[word for word in document.lower().split() 
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
texts = [[word for word in text 
			if word not in infrequent_words ]
			for text in texts]


dictionary = corpora.Dictionary.from_documents(texts)
# print(dictionary.id2token)

#dictionary.save('/tmp/newsgroups_train.dict') # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize('/tmp/newsgroups_train.mm', corpus) # store to disk, for later use
#print(corpus)

id_to_token = dict((id, token) for token, id in dictionary.token2id.iteritems())

lda = models.ldamodel.LdaModel(corpus=corpus, id2word=id_to_token, num_topics=20, 
		update_every=10, chunksize=100000, passes=5)

#lda.print_topics(20)
    




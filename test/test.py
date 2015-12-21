from __future__ import division
import re
import sys 
import pickle  
import nltk, nltk.classify.util, nltk.metrics
import collections
import string 
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from random import shuffle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 1521 Positive training 
# 13379 Negative Training 
# 507 Positive Testing 
# 4460 Negative Testing 


reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
positiveData = '../data/twssstories.txt'
negativeData1 = '../data/wikiquotes.txt'
negativeData2 = '../data/fml.txt'
negativeData3 = '../data/tflnonesent.txt'

oneliner_positive = '../data/Jokes16000.txt'
oneliner_negative = '../data/MIX16000.txt'

# for cross-validation
chunk_TrainingSet = []
chunk_TestingSet  = []
stopwords = [] 


def readFile(path): 
	f = open(path, 'r')
	text = f.read()
	f.close()
	return text.split('\n')

def read(path): 
	f = open(path, 'r')
	text = f.read()

	f.close()
	return text

def cleanData(line):
	stemmer = PorterStemmer()
#	snowball_stemmer = SnowballStemmer("english", ignore_stopwords = True)
#	wnl_stemmer = WordNetLemmatizer()
	no_stop = []
	for word in line.split():
		no_stop.append(str(stemmer.stem(word.lower())))
	remove_punct = ' '.join(word.strip(string.punctuation) for word in no_stop) 
	#stemmed = [stemmer.stem(p) for p in remove_punct]
	return remove_punct

# take in a some array of strings (each index has a sentence)
# return the first 3/4 of the array of strings for training data
# assumes everything is randomly sorted and there is no bias in terms of ordering
def getTraining(text): 
	length = len(text)
	numLines = float(length*3)/4
	return text[:int(numLines)]

# returns last 1/4 elements of the given array
def getTesting(text):
	length = len(text)
	numLines = float(length*3)/4
	return text[int(numLines):]

# add a "pos" or "neg" tag to the end of each sentence 
# this tag will allow us to train the classifier on positive & negative datasets
# eg. ("Put it back in, it's still moist in the middle.", 'pos')
# returns a tuple containing a "string" and a 'tag'
# change name - not necessarily positive
def tag_sentence(text, feature):
	data = []
	for t in text:
		tup = () 
		t = t.replace("'", "")
		t = cleanData(t) # ' '.join(word.strip(string.punctuation) for word in t.split()) 
		tup = (t, feature)
		data.append(tup)
	return data

# combine pos neg data
def load_data(file, tag, training_or_testing):
	txt = readFile(file)
	if training_or_testing is "training": 
		txt = getTraining(txt)
	if training_or_testing is "testing":
		txt = getTesting(txt)
	txt = tag_sentence(txt, tag) 
	return txt

# after getting the tagged and parsed data, 
# iterate through each line in the resulting txt file and 
# feed to the classifer. 
def train(training_data):
	training_feature_set = [(extract_features(line), label) for (line, label) in training_data]
	classifier = nltk.NaiveBayesClassifier.train(training_feature_set)
	return classifier

# we're currently training the classifier on unigrams 
def extract_features(phrase):
	words = nltk.word_tokenize(phrase)
	features = {}
#	bigrams = ngrams(words, 3)
#	for (w1, w2, w3) in bigrams:
#		features['contains(%s, %s, %s)' % (w1, w2, w3)] = ((w1, w2, w3) in bigrams)
	for word in words:
		features['contains(%s)' % word] = (word in words)
	return features

def save_classifier(data, filename ='classifier.dump'): 
	ofile = open(filename, 'w+')
	pickle.dump(data, ofile)
	ofile.close()

def test_classifier(classifier, testData):
	testing_feature_set = [(extract_features(line), label) for (line, label) in testData]
	return nltk.classify.accuracy(classifier, testing_feature_set)

def load_classifier(filename = 'classifier.dump'):
	ifile = open(filename, 'r+')
	classifier = pickle.load(ifile)
	ifile.close()
	return classifier

def save(data, filename):
	ofile = open(filename, "w+")
	pickle.dump(data, ofile)
	ofile.close()

def load(filename):
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

# takes in some text, shuffles, splits into pos and neg
def get_tagged(txt, tag):
	length = len(txt)
	numLines = float(length*3)/4
	training_data = txt[:int(numLines)]
	testing_data = txt[int(numLines):]
	training_data = tag_sentence(training_data, tag)
	testing_data = tag_sentence(testing_data, tag)
	return (training_data, testing_data)


def getData():
	pos_data = readFile(positiveData)
	shuffle(pos_data)
	oneliner_n = readFile(oneliner_negative)
	oneliner_p = readFile(oneliner_positive)
	neg_data_1 = readFile(negativeData1) # 97%
	neg_data_2 = readFile(negativeData2) # 83%
	neg_data_3 = readFile(negativeData3) # tflonesent 95.9% 
	neg_data = neg_data_1 + neg_data_2 + neg_data_3
	shuffle(neg_data)
	neg = []
	pos = []
	for n in neg_data: 
		neg.append(cleanData(n))
	for p in pos_data: 
		pos.append(cleanData(p))
	return (pos, neg)

def getMostCommon():
#	pos = read(positiveData)
#	neg1 = read(negativeData1)
#	neg2 = read(negativeData2)
#	neg3 = read(negativeData3)
	#alldata = pos + neg1 + neg2 + neg3
	alldata = read(oneliner_positive) + read(oneliner_negative)
	predicate = lambda x:x not in string.punctuation
	filter(predicate, alldata)
	all_data = nltk.tokenize.word_tokenize(alldata)
	all_data_distribution = nltk.FreqDist(w.lower() for w in all_data)
	most_common = all_data_distribution.most_common(50)
	d_common = dict(most_common)
	return d_common.keys()
	print "Top 10 most frequent words: ", d_common.keys()

# 1521 Positive training 
# 13379 Negative Training 
# 507 Positive Testing 
# 4460 Negative Testing 

def chunkifyData (text,chunks_list,text_length,chunk_length):
	for i in range(0,text_length,chunk_length):
		chunks_list.append(text[i:i+chunk_length])
	return chunks_list

def getChunkifiedData():
	pos_chunks = []
	neg_chunks = []
	
	pos_sizerange = 1521/4
	neg_sizerange = 13379/4

	pos_data = readFile(positiveData)
	neg_data_1 = readFile(negativeData1) 
	neg_data_2 = readFile(negativeData2) 
	neg_data_3 = readFile(negativeData3) 
	neg_data = neg_data_1 + neg_data_2 + neg_data_3

	pos_step = pos_sizerange%1521
	neg_step = neg_sizerange%13379

	pos_data = tag_sentence(pos_data,"pos")
	neg_data = tag_sentence(neg_data,"neg")

	chunkifyData(pos_data, pos_chunks, 1521, pos_step)
	chunkifyData(neg_data, neg_chunks, 13379, neg_step)

	shuffle(pos_chunks)
	shuffle(neg_chunks)

	#after every shuffle the first index has a different piece of the text, so we test on the other 4 indeces
	chunk_TrainingSet.extend(pos_chunks[0] + neg_chunks[0] )
	chunk_TestingSet.extend(pos_chunks[0]  + neg_chunks[0] + pos_chunks[1] + neg_chunks[1] + pos_chunks[2] + neg_chunks[2] + pos_chunks[3] + neg_chunks[3] + pos_chunks[4] + neg_chunks[4])

	return (chunk_TrainingSet, chunk_TestingSet)

def precision_recall(classifier, testing_data):
	errors = [] 
	true_pos = 0
	all_retrieved = 0
	all_pos = 0
	for (prompt, tag) in testing_data:
		guess = classifier.classify(extract_features(prompt))
		if (guess == tag) and (guess == "pos"):
			true_pos+= 1
		if guess == "pos":
			all_retrieved+=1
		if tag == "pos":
			all_pos+=1
	print "true pos: ", true_pos
	print "all retrieved: ", all_retrieved
	print "recall: ", all_pos
	precision = float(true_pos/all_retrieved)
	recall = float(true_pos/all_pos)
	return (precision, recall)
# def cross_validate(testing_chunk,):
# 	getChunkifiedData()

# 	test_classifier(classifier, shuffled_testing_data)

def save_to_file(text, filename):
	f = open(filename, 'w')
	f.write('\n'.join(str(v) for v in text))
	f.close()


stopwords = getMostCommon()
#get_data = getData()

all_data = getData()

save(all_data, "data.dump")

#all_data = load("data.dump")
pos_data = get_tagged(all_data[0], "pos")
neg_data = get_tagged(all_data[1], "neg")
# train on just one negative dataset - test with the remaining 1/4th of the data set + all others
# 94.88 (negative 1)
# 94.1 (neg 2)
# 93 (neg 3)
# neg1[0] -> training; neg1[1] -> testing
training_data = pos_data[0] + neg_data[0] 
testing_data = pos_data[1] + neg_data[1]

#getChunkifiedData()

#save_to_file(training_data, 'training_data.txt')
#save_to_file(testing_data, 'testing_data.txt')

# train and save data - training data takes a while 
data = train(training_data)
#data = train(chunk_TrainingSet)
save_classifier(data)
classifier = load_classifier()
print test_classifier(classifier, testing_data)
errors = []
for (prompt, tag) in testing_data:
	guess = classifier.classify(extract_features(prompt))
	if guess != tag:
		errors.append( (tag, guess, prompt) )

#for (tag, guess, name) in sorted(errors):
#	print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))

# tests classifier, prints out the accuracy 
#print test_classifier(classifier, testing_data)
# prints out the most informative features (which words had the biggest impact)
print classifier.show_most_informative_features()
analysis = precision_recall(classifier, testing_data)

print "precision: ", analysis[0]
print "recall: ", analysis[1]


# NO STEM + STOPWORD 
# accuracy = 94.966% accuracy 
# Recall - 58%; Precision - 88% 

# STEMMING + STOPWORDING 
# Accuracy = 95.2% 
# Recall 56%; Precision 93% 

# NO STEMMING + NO STOPWORDING 
# Accuracy 95.3% 
# Precision 89% 
# Recall: 61.5%







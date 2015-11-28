import re
import sys 
import pickle  
import nltk, nltk.classify.util, nltk.metrics
import collections
import string 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from random import shuffle

# don't actually need this anymore
from textblob.classifiers import NaiveBayesClassifier


reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
positiveData = '../data/twssstories.txt'
negativeData1 = '../data/wikiquotes.txt'
negativeData2 = '../data/fml.txt'
negativeData3 = '../data/tflnonesent.txt'


def readFile(path): 
	f = open(path, 'r')
	text = f.read()
	f.close()
	return text.split('\n')

def cleanData(line):
	no_stop = []
	for word in line.split():
		if word not in stopwords.words('english'):
			no_stop.append(word.lower())
	remove_punct = ' '.join(word.strip(string.punctuation) for word in no_stop) 
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
	new_features = {}
	for word in words:
		features['contains(%s)' % word] = (word in words)
	return features

def save(data, filename ='classifier.dump'): 
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

# takes in some text, shuffles, splits into pos and neg
def getShuffledData(txt, tag):
	shuffle(txt)
	length = len(txt)
	numLines = float(length*3)/4
	training_data = txt[:int(numLines)]
	testing_data = txt[int(numLines):]
	training_data = tag_sentence(training_data, tag)
	testing_data = tag_sentence(testing_data, tag)
	return (training_data, testing_data)

def getData():
	pos_data = readFile(positiveData)
	neg_data_1 = readFile(negativeData1)
	neg_data_2 = readFile(negativeData2)
	neg_data_3 = readFile(negativeData3)
	neg_data = neg_data_1 + neg_data_2 + neg_data_3
	return (pos_data, neg_data)

def save_to_file(text, filename):
	f = open(filename, 'w')
	f.write('\n'.join(str(v) for v in text))
	f.close()


all_data = getData()
pos_data = getShuffledData(all_data[0], "pos")
neg_data = getShuffledData(all_data[1], "neg")

shuffled_training_data = pos_data[0] + neg_data[0] 
shuffled_testing_data = pos_data[1] + neg_data[1]

save_to_file(shuffled_training_data, 'training_data.txt')
save_to_file(shuffled_testing_data, 'testing_data.txt')

# train and save data - training data takes a while 
#data = train(shuffled_testing_data)
#save(data)
classifier = load_classifier()

errors = [] 
for (prompt, tag) in shuffled_testing_data:
	guess = classifier.classify(extract_features(prompt))
	if guess != tag:
		errors.append((tag, guess, prompt))

for (tag, guess, name) in sorted(errors):
	print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))

# tests classifier, prints out the accuracy 
print test_classifier(classifier, shuffled_testing_data)
# prints out the most informative features (which words had the biggest impact)
print classifier.show_most_informative_features()


# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)
 
# for i, (feats, label) in enumerate(testing_data):
# 	refsets[label].add(i)
# 	observed = classifier.classify(extract_features(feats))
# 	testsets[observed].add(i)

#print extract_features("Ahh, it hit me right in the face.")
#print cleanData("Ahh, it hit me right in the face.")
#print loadTestData(negativeData1)
#print test(data, "Ahh, it hit me right in the face.")
# # some docs are longer than others, 
# # use tf_idf to downscale the weights of teh words that occur in many docs 
# # and are therefore less informative



# write results to a doc (for debugging)
#f = open('parsedData.txt', 'w')
#f.write('\n'.join(str(v) for v in trainingData))
#f.close()

# def getData(training_or_testing):
# 	data = load_data(negativeData1, "neg", training_or_testing)
# 	data += load_data(negativeData2, "neg", training_or_testing)
# 	data += load_data(negativeData3, "neg", training_or_testing)
# 	print 'negative data: ', len(data), ' ', training_or_testing
# 	data += load_data(positiveData, "pos", training_or_testing)
# 	print 'positive data: ', len(data), ' ', training_or_testing
# 	return data


import re
import sys 
import pickle  
import nltk
from nltk.tokenize import word_tokenize
import string 
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
	return ' '.join(word.strip(string.punctuation) for word in line.split()) 

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
		t = ' '.join(word.strip(string.punctuation) for word in t.split()) 
		tup = (t, feature)
		#data[t] = feature
		data.append(tup)
	return data

# combine pos neg data
def load_data(file, tag, training_or_testing):
	txt = readFile(file)
	if training_or_testing is "training": 
		txt = getTraining(txt)
	if training_or_testing is "testing":
		txt = getTesting(txt)
	#txt = getTraining(txt)
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

def getData(training_or_testing):
	data = load_data(negativeData1, "neg", training_or_testing)
	data += load_data(positiveData, "pos", training_or_testing)
	data += load_data(negativeData2, "neg", training_or_testing)
	data += load_data(negativeData3, "neg", training_or_testing)
	return data

# load data, parse and tag
training_data = getData("training")
len(training_data)
testing_data = getData("testing")
len(testing_data)

# train and save data 
# data = train(trainingData)
# save(data)
classifier = load_classifier()
print test_classifier(classifier, testing_data)
print classifier.show_most_informative_features()
#print loadTestData(negativeData1)
#print test(data, "Ahh, it hit me right in the face.")
# # some docs are longer than others, 
# # use tf_idf to downscale the weights of teh words that occur in many docs 
# # and are therefore less informative



# write results to a doc (for debugging)
#f = open('parsedData.txt', 'w')
#f.write('\n'.join(str(v) for v in trainingData))
#f.close()


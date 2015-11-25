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

def cleanData(text):
	for t in text:
		text[t] = re.compile("[^\w]")
	return text 

# take in a some array of strings (each index has a sentence)
# return the first 3/4 of the array of strings for training data
# assumes everything is randomly sorted and there is no bias in terms of ordering
def getTraining(text): 
	length = len(text)
	numLines = float(length*3)/4
	return text[:int(numLines)]

# add a "pos" or "neg" tag to the end of each sentence 
# this tag will allow us to train the classifier on positive & negative datasets
# eg. ("Put it back in, it's still moist in the middle.", 'pos')
# returns a tuple containing a "string" and a 'tag'
# change name - not necessarily positive
def tagPos(text, feature):
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
def loadData(file, tag):
	txt = readFile(file)
	txt = getTraining(txt)
	txt = tagPos(txt, tag) 
	return txt

# after getting the tagged and parsed data, 
# iterate through each line in the resulting txt file and 
# feed to the classifer. 
def train(training_data):
	training_feature_set = [(extract_features(line), label) for (line, label) in training_data]
	#cl = NaiveBayesClassifier(pos_neg_text)
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

# load data, parse and tag
trainingData = loadData(positiveData, "pos")
trainingData += loadData(negativeData1, "neg")
trainingData += loadData(negativeData2, "neg")
trainingData += loadData(negativeData3, "neg")

# write results to a doc (for debugging)
f = open('parsedData.txt', 'w')
f.write('\n'.join(str(v) for v in trainingData))
f.close()

# train and save data 
data = train(trainingData)
save(data)

# # some docs are longer than others, 
# # use tf_idf to downscale the weights of teh words that occur in many docs 
# # and are therefore less informative

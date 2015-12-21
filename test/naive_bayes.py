import pickle  
import nltk 
from nltk.stem.porter import *
import string 
import sys 

###### Naive Bayes #######
# Trains a naive bayes classifier on data 

reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
# saves data to a pickle file 
def save(data, filename):
	""" Saves data in a pickle file """
	ofile = open(filename, "w+")
	pickle.dump(data, ofile)
	ofile.close()

# helper fuction: loads data from a pickle file
def load(filename):
	""" Loads and returns data from pickle file """
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

# add a "pos" or "neg" tag to the end of each sentence 
# this tag will allow us to train the classifier on positive & negative datasets
# eg. ("Put it back in, it's still moist in the middle.", 'pos')
# returns a tuple containing a "string" and a 'tag'
# change name - not necessarily positive
def tag_sentence(text, feature):
	""" Tag each prompt in the data as either an example of a positive or a negative twss prompt. 
	This is achieved by creating tuples in place of prompts in the format ("prompt", "tag"). The tag 
	is simply "pos" or "neg". Returns the tagged dataset """  
	data = []
	for t in text:
		tup = () 
		t = t.replace("'", "")
		tup = (t, feature)
		data.append(tup)
	return data

# after getting the tagged and parsed data, 
# iterate through each line in the resulting txt file and 
# feed to the classifer. 
def train(training_data):
	""" Train the Naive Bayes Classifier on unigrams """ 
	training_feature_set = [(extract_features(line), label) for (line, label) in training_data]
	classifier = nltk.NaiveBayesClassifier.train(training_feature_set)
	return classifier

def test_classifier(classifier, testData):
	""" Test the effectiveness of the classifier. Return the overall accuracy of the classifier on the testing data """ 
	testing_feature_set = [(extract_features(line), label) for (line, label) in testData]
	return nltk.classify.accuracy(classifier, testing_feature_set)

# we're currently training the classifier on unigrams 
def extract_features(phrase):
	""" Returns unigrams in the prompt. """ 
	words = nltk.word_tokenize(phrase)
	features = {}
	for word in words:
		features['contains(%s)' % word] = (word in words)
	return features

def get_errors(testing_data, classifier): 
	""" Returns false positives and false negatives """ 
	errors = []
	for (prompt, tag) in testing_data:
		guess = classifier.classify(extract_features(prompt))
		if guess != tag:
			errors.append( (tag, guess, prompt) )
	for (tag, guess, name) in sorted(errors):
		print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))

# NLTK doesn't seem to have a built in precision/recall function so I'm just writing my own 
def get_precision_recall(classifier, testing_data):
	""" Classify each prompt in the testing data invidually and count the number of
	true positives, all retrieved, and all positives. 
	Precision = true positive / all retrieved 
	Recall = true positive / all positive. """  
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
	print 'true pos: ', true_pos
	print 'all retrieved: ', all_retrieved
	print 'all positive: ', all_pos
	precision = float(true_pos/all_retrieved)
	print "Precision: ", precision
	recall = float(true_pos/all_pos)
	print "Recall: ", recall 
	return (precision, recall)

# load data
pos_training = load("pos_data_training.dump")
neg_training = load("neg_data_training.dump")
pos_testing = load("pos_data_testing.dump")
neg_testing = load("neg_data_testing.dump")
# tag sentences as negative 
pos_training = tag_sentence(pos_training, "pos")
neg_training = tag_sentence(neg_training, "neg")
# train 
nb_classifier = train(pos_training + neg_training)
save(nb_classifier, 'nb_classifier.dump')

# testing 
pos_testing = tag_sentence(pos_testing, "pos")
neg_testing = tag_sentence(neg_testing, "neg")
testing_data = pos_testing + neg_testing 
print test_classifier(nb_classifier, testing_data)
#get_errors(testing_data, nb_classifier)
print get_precision_recall(nb_classifier, testing_data)





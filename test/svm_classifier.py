""" Trains and tests SVM Classifier on TWSS Data """

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

import pickle  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import sys
from sklearn import metrics

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

def get_classifier(): 
	""" Creates and returns a pipeline with a CountVectorizer and SVC classifier """
	text_clf = Pipeline([
		('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
		('clf', SVC(kernel='rbf')),
		#('clf', MultinomialNB()),
		])
	return text_clf

def train(classifier, training_data, target):
	""" Return classifier trained on training data and target """ 
	classifier.fit(training_data, target)
	return classifier
# assume that the pos and neg arrays never change positions after this 
# cuz then you'd be fucked. 
# use a dictionary? 
def get_target(pos_data, neg_data):
	""" Returns the target vector: positive prompts are marked as 0, negative prompts are marked as 1 """ 
	pos_target = [0]*len(pos_data)
	neg_target = [1]*len(neg_data)
	return pos_target+neg_target

def print_metrics(trained_classifier, testing_data):
	predicted = trained_classifier.predict(testing_data)
	print np.mean(predicted == testing_target)
# load data
pos_training = load("pos_data_training.dump")
neg_training = load("neg_data_training.dump")

pos_testing = load("pos_data_testing.dump")
neg_testing = load("neg_data_testing.dump")

# get testing target
testing_data = pos_testing + neg_testing
testing_target = get_target(pos_testing, neg_testing)

# get training target
training_data = pos_training + neg_training
training_target = get_target(pos_training, neg_training)

# load and train classifier
classifier = get_classifier() 
trained_classifier =  train(classifier, training_data, training_target)

# pickling erorr? 
#save(trained_classifier, "svm_classifier.pk")

predicted = trained_classifier.predict(testing_data)
 
print np.mean(predicted == testing_target)
print (metrics.classification_report(testing_target, predicted))


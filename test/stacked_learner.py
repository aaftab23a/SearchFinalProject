from __future__ import division
import re
import sys 
import pickle  
import collections
import string 

import numpy as np
import treetaggerwrapper

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
#from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm, datasets
import os

# set working directories
MY_DIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_DIR, 'PickledData')

reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
# load the dataset
positiveData = '../data/twssstories.txt'
negativeData2 = '../data/wikiquotes.txt'
negativeData1 = '../data/fml.txt'
negativeData3 = '../data/tflnonesent.txt'

training_data = []
testing_data  = [] 

def read_file(path) : 
	f = open(path, 'r')
	text = f.read()
	f.close()
	# remove punctuation
	exclude = set(string.punctuation)
	text = ''.join(ch for ch in text if ch not in exclude)

	# turn uppercase to lowercase
	text = text.lower()
	return text.split('\n')

# we load the output of the classifier to a pickle file
def load(filename):
	""" Loads and returns data from pickle file """
	fname = os.path.join(PICKLE_DIR, filename)
	with open(fname, 'r+') as fp:
		data = pickle.load(fp)
	return data

# tokenize the text
def process_text(txt) :
	text =[]
	for t in txt:
		text.append(word_tokenize(t))
	return text

# make training dataset
pos_data = read_file(positiveData)
neg_data1 = read_file(negativeData1)
neg_data2 = read_file(negativeData2)
# include the 2 negative data sources
neg_data = neg_data1 + neg_data2
# take the first 2020 lines
neg_data = neg_data[:2020]

# instantiate the tagger
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')

def convert_to_unicode(txt):
	"""The POS treetagger needs all the characters to be converted to unicode"""
	u_txt = []
	for t in txt :
		u_txt.append(unicode(t,"utf-8"))
	return u_txt

def tag_text (txt) :
	tags = []
	for t in txt:
		tags.append(tagger.tag_text(t))
	return tags

# convert all the training datasets
pos_unicode_text = convert_to_unicode(pos_data)
neg_unicode_text = convert_to_unicode(neg_data)

def extract_tags (tags) :
	"""From the POS tagging output, we only want the sequence of tags"""
	extracted = []
	for string in tags :
		st = ""
		for term in string : 
			st += term.split()[1]+ " "
		extracted.append(st)
	return extracted
# these are the split tags of the positive & negative datasets

pos_tags = tag_text(pos_unicode_text)
neg_tags = tag_text(neg_unicode_text)

# we are training on the extracted sequence of tags
final_pos_data = extract_tags(pos_tags)
final_neg_data = extract_tags(neg_tags)

# we are using the same ratioof positive to negative training data
training_data = final_pos_data[:1515]+final_neg_data[:1515]
testing_data  = final_pos_data[1516:2020]+final_neg_data[1516:2020]


# this is the svm classifier we are using for content based features
text_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
            	 	('clf', SVC(kernel='rbf')),
        ])

##########################  POS tagging training & testing  ##########################  
# populating the training vectors
training_pos_target = [0]*len(final_pos_data[:1515])
training_neg_target = [1]*len(final_neg_data[:1515])
# populating the testing vectors
testing_pos_target = [0]*len(final_pos_data[1516:2020])
testing_neg_target = [1]*len(final_neg_data[1516:2020])
# combining the positive and negative datasets
training_target = training_pos_target + training_neg_target
testing_target = testing_pos_target + testing_neg_target
# train the POS svm classifier
text_clf = text_clf.fit(training_data, training_target)

pos_predicted = text_clf.predict(testing_data)

# print the output accuracy, precision and recall of the POS tagging model
print "##################################### POS tagging ####################################"
print np.mean(pos_predicted == testing_target)
print (metrics.classification_report(testing_target, pos_predicted))
print metrics.confusion_matrix(testing_target, pos_predicted)

# save the classifier output toa pickle file
def save_classifier_output ( data, file_name):
	ofile = open(file_name, 'w+')
	output = []
	for i in data : 
		output.append(i)

	pickle.dump(output, ofile)
	ofile.close()

save_classifier_output( pos_predicted,"pos_output.dump" )

##########################  Stacked learner training & testing  ########################## 
# our 3 models
# pos tagging => svm
# content based => naive bayse
# content based => DecisionTree

# we load the output of the 3 classifiers into the 3 variables
pos_output  = load("pos_output.dump")
svm_output  = load("svm_output.dump")
tree_output = load("tree_output.dump")

def find_average_prediction() :
	"""walks through the prediction vectors and finds the mean prediction of each prompt"""
	# cases
		# 0 0 1 =>0
		# 1 1 0 =>1
		# 0 0 0 =>0
		# 1 1 1 =>1
	avg_output = []
	for x , y, z in zip( pos_output, svm_output, tree_output) :
		sum_tag = x + y + z
		if sum_tag == 0 or sum_tag == 1:
			avg_output.append(0)
		elif sum_tag >= 2 :
			avg_output.append(1)
	return avg_output

# load the testing data
pos_data = load("pos_data_training.dump")
neg_data = load("neg_data_training.dump")

testing_pos = pos_data[1516:2020]
testing_neg = neg_data[1516:2020]
# make pos&neg testing data that is equal in length to the training one
testing_pos_target = [0]*len(testing_pos)
testing_neg_target = [1]*len(testing_neg)

testing_target = testing_pos_target + testing_neg_target

average_predicted = find_average_prediction()

# print the output accuracy, precision and recall of the stacked model
print "################################## Stacked model ################################## "
np.array(average_predicted).dump(open('avg.npy', 'wb'))
avg = np.load(open('avg.npy', 'rb'))

print np.mean( avg == testing_target )
print (metrics.classification_report(testing_target, avg))
print metrics.confusion_matrix(testing_target, avg)











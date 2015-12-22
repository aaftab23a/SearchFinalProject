""" Trains data using the Random Forest Classifier """
__author__      = "Nachwa El khamlichi"
__email__ = "elkha22n@mtholyoke.edu"

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import os, sys
from sklearn import metrics
import numpy as np
import pickle  

# set working directories
MY_DIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_DIR, 'PickledData')

# instantiate the random forest classifier
tree_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
            	 	('clf', RandomForestClassifier()), 
        ])
 
def load(filename):
	""" Loads and returns data from pickle file """
	fname = os.path.join(PICKLE_DIR, filename)
	with open(fname, 'r+') as fp:
		data = pickle.load(fp)
	return data

# build the training and testing datasets
pos_data = load("pos_data_training.dump")
neg_data = load("neg_data_training.dump")

training_pos = pos_data[:1515]
training_neg = neg_data[:1515]

testing_pos = pos_data[1516:2020]
testing_neg = neg_data[1516:2020]

training_pos_target = [0]*len(training_pos)
training_neg_target = [1]*len(training_neg)

testing_pos_target = [0]*len(testing_pos)
testing_neg_target = [1]*len(testing_neg)


training_target = training_pos_target + training_neg_target
training_data   = training_pos + training_neg
testing_target	= testing_pos_target + testing_neg_target
testing_data 	= testing_pos + testing_neg

tree_clf = tree_clf.fit(training_data, training_target)
tree_predicted = tree_clf.predict(testing_data)

print "##################################### RandomForest training & testing###########################"
print np.mean(tree_predicted == testing_target)
print (metrics.classification_report(testing_target, tree_predicted))
print metrics.confusion_matrix(testing_target, tree_predicted)

def save_classifier_output ( data, file_name):
	''' Pickle data and save '''
 	ofile = open(file_name, 'w+')
 	output = []
 	for i in data : 
 		output.append(i)
 	pickle.dump(output, ofile)
 	ofile.close()
# save the prediction vector to a dump file
save_classifier_output( tree_predicted,"tree_output.dump") 


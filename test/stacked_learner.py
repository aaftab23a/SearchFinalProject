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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm, datasets
import os


reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
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


def process_text(txt) :
	text =[]
	for t in txt:
		text.append(word_tokenize(t))
	# print text
	return text

def pos_tagging (txt) :
	tples =[]
	for t in txt :
		tples.append(nltk.pos_tag(t))
	# print tples
	return tples

pos_data = read_file(positiveData)

neg_data1 = read_file(negativeData1)
neg_data2 = read_file(negativeData2)
# neg_data3 = read_file(negativeData3)

neg_data = neg_data1 + neg_data2

neg_data = neg_data[:2020]
print 'neg data ', len(neg_data)

# len = 5968

def clean_tuples(text) :
	temp = ''
	data = []
	for t in text:
		for p in tples:
			temp += p[1] +''
		data.append(temp)
	print data 
	return data

def convert_to_unicode(txt):
	u_txt = []
	for t in txt :
		u_txt.append(unicode(t,"utf-8"))
	return u_txt

tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
txt = ["this is a very short text to tag","this is another text to tag"]


pos_unicode_text = convert_to_unicode(pos_data)
neg_unicode_text = convert_to_unicode(neg_data)
unicode_text 	 = convert_to_unicode(txt)

def extract_tags (tags) :
	extracted = []
	for string in tags :
		st = ""
		for term in string : 
			st += term.split()[1]+ " "
		extracted.append(st)
	return extracted


def tag_text (txt) :
	tags = []
	for t in txt:
		tags.append(tagger.tag_text(t))
	return tags

tags = tag_text(unicode_text)


pos_tags = tag_text(pos_unicode_text)
neg_tags = tag_text(neg_unicode_text)

final_pos_data = extract_tags(pos_tags)
final_neg_data = extract_tags(neg_tags)

training_data = final_pos_data[:1515]+final_neg_data[:1515]
testing_data  = final_pos_data[1516:2020]+final_neg_data[1516:2020]


text_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
            	 	('clf', SVC(kernel='rbf')),
        ])

def load(filename):
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

######################## pos train/test data #######################
training_pos_target = [0]*len(final_pos_data[:1515])
training_neg_target = [1]*len(final_neg_data[:1515])

testing_pos_target = [0]*len(final_pos_data[1516:2020])
testing_neg_target = [1]*len(final_neg_data[1516:2020])

training_target = training_pos_target + training_neg_target
testing_target = testing_pos_target + testing_neg_target

text_clf = text_clf.fit(training_data, training_target)

pos_predicted = text_clf.predict(testing_data)

print "##################################### POS tagging ###########################"
print np.mean(pos_predicted == testing_target)
print (metrics.classification_report(testing_target, pos_predicted))
print metrics.confusion_matrix(testing_target, pos_predicted)


def save_classifier_output ( data, file_name):
	ofile = open(file_name, 'w+')
	output = []
	for i in data : 
		output.append(i)

	pickle.dump(output, ofile)
	ofile.close()

save_classifier_output( pos_predicted,"pos_output.dump" )


# our 3 models
# pos tagging => svm
# content based => naive bayse
# content based => DecisionTree

pos_output  = load("pos_output.dump")
svm_output   = load("svm_output.dump")
tree_output = load("tree_output.dump")

if collections.Counter(pos_output) == collections.Counter(svm_output) :
	print "******************** they are the same ***************"

# cases
# 0 0 1 
# 1 1 0 
# 0 0 0 
# 1 1 1

def find_average_prediction() :
	avg_output = []
	for x , y, z in zip( pos_output, svm_output, tree_output) :
		sum_tag = x + y + z
		if sum_tag == 0 or sum_tag == 1:
			avg_output.append(0)
		elif sum_tag >= 2 :
			avg_output.append(1)
	return avg_output

all_data = load("data.dump")
pos_data = all_data[0]
neg_data = all_data[1]

testing_pos = pos_data[1516:2020]
testing_neg = neg_data[1516:2020]

testing_pos_target = [0]*len(testing_pos)
testing_neg_target = [1]*len(testing_neg)

testing_target = testing_pos_target + testing_neg_target

average_predicted = find_average_prediction()

print "################################## avg #########################################"
np.array(average_predicted).dump(open('avg.npy', 'wb'))
avg = np.load(open('avg.npy', 'rb'))

print np.mean( avg == testing_target )
print (metrics.classification_report(testing_target, avg))
print metrics.confusion_matrix(testing_target, avg)











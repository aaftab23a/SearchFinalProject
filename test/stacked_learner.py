from __future__ import division
import multiprocessing as mp
import re
import sys 
import pickle  
import collections
import string 
import pprint

import numpy as np

import treetaggerwrapper
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from random import shuffle

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
testing_data = [] 

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
neg_data = read_file(negativeData1)

# pos_tagging(pos_data)
# pos_tagging(neg_data)


# def clean_tuples(tples) :
# 	temp = []
# 	for t in tples:
# 		# print t
# 		temp.append(t[1])
# 	print temp
# 	return temp

def clean_tuples(text) :
	temp = ''
	data = []
	for t in text:
		print t
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
	print extracted
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

training_data = final_pos_data[:1515]+final_neg_data[:1326]
testing_data  = final_pos_data[1516:2020]+final_neg_data[1327:]


text_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
            	 	('clf', DecisionTreeClassifier()), 
        ])

training_pos_target = [0]*len(final_pos_data[:1515])
training_neg_target = [1]*len(final_neg_data[:1326])

testing_pos_target = [0]*len(final_pos_data[1516:2020])
testing_neg_target = [1]*len(final_neg_data[1327:])


training_target = training_pos_target + training_neg_target
testing_target = testing_pos_target + testing_neg_target


text_clf = text_clf.fit(training_data, training_target)

predicted = text_clf.predict(testing_data)
print np.mean(predicted == testing_target)
print (metrics.classification_report(testing_target, predicted))
print metrics.confusion_matrix(testing_target, predicted)









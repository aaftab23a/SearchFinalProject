from __future__ import division
import multiprocessing as mp
import re
import sys 
import pickle  
import nltk, nltk.classify.util, nltk.metrics
import collections
import string 
import pprint

# import sklearn.datasets
# import numpy as np

import treetaggerwrapper
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from random import shuffle

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline


import os

reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
positiveData = '../data/twssstories.txt'
negativeData1 = '../data/wikiquotes.txt'
negativeData2 = '../data/fml.txt'
negativeData3 = '../data/tflnonesent.txt'

stopwords = [] 

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

extract_tags(pos_tags)
extract_tags(neg_tags)


# tags2 = treetaggerwrapper.make_tags(tags)
# pprint.pprint( tags2)



# if __name__ == '__main__':
#     p = Process(target=clean_tuples, args=(pos_tagging(d[:2]),pos_tagging(d[2:])))
#     p.start()
#     p.join()

# processes = [mp.Process(target=clean_tuples, args=(pos_tagging(d))) for x in range(4)]

# # Run processes
# for p in processes:
#     p.start()

# # Exit the completed processes
# for p in processes:
#     p.join()













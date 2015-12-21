############## Process Data ################
# Reads, processes and saves data as positive training, negative training 
# and positive testing and negative testing 
# pos_data_training 
# pos_data_testing 
# neg_data_training 
# neg_data_testing 


import sys 
import string 
from random import shuffle
import pickle  
from nltk.stem.porter import *


reload(sys) 
sys.setdefaultencoding('ISO-8859-1')
positiveData = '../data/twssstories.txt'
negativeData1 = '../data/wikiquotes.txt'
negativeData2 = '../data/fml.txt'
negativeData3 = '../data/tflnonesent.txt'


# reads file from path
# returns a string of text
def readFile(path): 
	""" Reads a file at the given path. Returns a string containing text in that file. """ 
	f = open(path, 'r')
	text = f.read()
	f.close()
	return text

# saves data to a pickle file 
def save(data, filename):
	""" Saves data in a pickle file """
	ofile = open(filename, "w+")
	pickle.dump(data, ofile)
	ofile.close()

# helper fuction: loads data from a pickle file
def load(filename):
	""" Saves data in a pickle file """
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

# removes stopwords, stems, removes punctuation
def process_data(line):
	""" Removes punctuation, stems words using the PorterStemmer, removes stopwords """ 
	stemmer = PorterStemmer()
	no_stop = []
	for word in line.split():
		no_stop.append(str(stemmer.stem(word.lower())))
	remove_punct = ' '.join(word.strip(string.punctuation) for word in no_stop) 
	return remove_punct

def getMostCommon(data):
	""" Finds and returns a list of the most commonly occuring terms in the dataset """ 
	predicate = lambda x:x not in string.punctuation
	filter(predicate, alldata)
	data = nltk.tokenize.word_tokenize(alldata)
	all_data_distribution = nltk.FreqDist(w.lower() for w in data)
	most_common = all_data_distribution.most_common(50)
	d_common = dict(most_common)
	return d_common.keys()
	print "Top 10 most frequent words: ", d_common.keys()

def get_data():
	""" Retrive, shuffles and cleans data. Returns a tuple in the form (positive data, negative data) """ 
	pos_data = readFile(positiveData).split("\n")
	neg_data_1 = readFile(negativeData1).split("\n") # 97%
	neg_data_2 = readFile(negativeData2).split("\n") # 83%
	neg_data_3 = readFile(negativeData3).split("\n") # tflonesent 95.9% 
	neg_data = neg_data_1 + neg_data_2 + neg_data_3
	shuffle(pos_data)
	shuffle(neg_data)
	return (clean_data(pos_data), clean_data(neg_data))

# Returns an array of cleandata
def clean_data(data):
	""" Cleans each element in the data object. """ 
	clean_data = [] 
	for d in data:
		clean_data.append(process_data(d))
	return clean_data

# take in a some array of strings (each index has a sentence)
# return the first 3/4 of the array of strings for training data
# assumes everything is randomly sorted and there is no bias in terms of ordering
def split_training(data):
	""" Splits data into training and testing. Training data consists of the first 3/4ths of a shuffled dataset. The remaining 1/4th is testing data. 
		Returns a tuple in the form (training data, testing data). """ 
	length = len(data)
	numLines = float(length*3)/4
	return (data[:int(numLines)], data[int(numLines):])

all_data = get_data()
save(all_data, "clean_data.dump")
split_pos_data = split_training(all_data[0])
split_neg_data = split_training(all_data[1])

save(split_pos_data[0], "pos_data_training.dump")
save(split_pos_data[1], "pos_data_testing.dump")
save(split_neg_data[0], "neg_data_training.dump")
save(split_neg_data[1], "neg_data_testing.dump")

print len(split_pos_data[0])
print len(split_neg_data[0])


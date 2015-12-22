''' Trains an nltk naive bayes classifier on data '''
import pickle  
import nltk 
import string 
import os, sys 


reload(sys) 
sys.setdefaultencoding('ISO-8859-1')

# lists used for cross-validation
chunk_TrainingSet = []
chunk_TestingSet  = []

# set working directories 
MY_DIR = os.path.realpath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(MY_DIR, 'PickledData')

def save(data, filename):
	""" Saves data in a pickle file """
	fname = os.path.join(PICKLE_DIR, filename)
	with open(fname, 'wb') as f:
		pickle.dump(data, f)

def load(filename):
	""" Loads and returns data from pickle file """
	fname = os.path.join(PICKLE_DIR, filename)
	with open(fname, 'r+') as fp:
		data = pickle.load(fp)
#	ifile = open(filename, 'r+')
#	data = pickle.load(ifile)
#	ifile.close()
	return data


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

def train(training_data):
	""" Train the Naive Bayes Classifier on unigrams """ 
	training_feature_set = [(extract_features(line), label) for (line, label) in training_data]
	classifier = nltk.NaiveBayesClassifier.train(training_feature_set)
	return classifier

def test_classifier(classifier, testData):
	""" Test the effectiveness of the classifier. Return the overall accuracy of the classifier on the testing data """ 
	testing_feature_set = [(extract_features(line), label) for (line, label) in testData]
	return nltk.classify.accuracy(classifier, testing_feature_set)

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
		# if true positive 
		if (guess == tag) and (guess == "pos"):
			true_pos+= 1
		if guess == "pos":
			all_retrieved+=1
		if tag == "pos":
			all_pos+=1
	precision = true_pos/float(all_retrieved)
	recall = true_pos/float(all_pos)
	return (precision, recall)

# 1521 Positive training 
# 13379 Negative Training 
# 507 Positive Testing 
# 4460 Negative Testing 
def chunkifyData (text,chunks_list,text_length,chunk_length):
	for i in range(0,text_length,chunk_length):
		chunks_list.append(text[i:i+chunk_length])
	return chunks_list

def getChunkifiedData(pos_data, neg_data):
	'''Performs 5-fold cross validation'''
	pos_chunks = []
	neg_chunks = []
	
	pos_sizerange = 1521/4
	neg_sizerange = 13379/4

	pos_step = pos_sizerange%1521
	neg_step = neg_sizerange%13379

	pos_data = tag_sentence(pos_data,"pos")
	neg_data = tag_sentence(neg_data,"neg")

	chunkifyData(pos_data, pos_chunks, 1521, pos_step)
	chunkifyData(neg_data, neg_chunks, 13379, neg_step)

	shuffle(pos_chunks)
	shuffle(neg_chunks)

	#after every shuffle the first index has a different piece of the text, so we test on the other 4 indeces
	chunk_TrainingSet.extend(pos_chunks[0] + neg_chunks[0] )
	chunk_TestingSet.extend(pos_chunks[0]  + neg_chunks[0] + pos_chunks[1] + neg_chunks[1] + pos_chunks[2] + neg_chunks[2] + pos_chunks[3] + neg_chunks[3] + pos_chunks[4] + neg_chunks[4])

	return (chunk_TrainingSet, chunk_TestingSet)


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
# get_errors(testing_data, nb_classifier)
print get_precision_recall(nb_classifier, testing_data)





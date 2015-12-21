import pickle  
import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#from sklearn import svm, datasets
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
#from sklearn.preprocessing import label_binarize

def load(filename):
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

# take in a some array of strings (each index has a sentence)
# return the first 3/4 of the array of strings for training data
# assumes everything is randomly sorted and there is no bias in terms of ordering
def getTraining(text): 
	length = len(text)
	numLines = float(length*3)/4
	return text[:int(numLines)]

# returns last 1/4 elements of the given array
def getTesting(text):
	length = len(text)
	numLines = float(length*3)/4
	return text[int(numLines):]
# add a shuffle function somewhere here
# split into training and testing data 
all_data = load("data.dump")
training_data = []
testing_data = [] 
pos_data = all_data[0]
neg_data = all_data[1]

training_pos = getTraining(pos_data)
training_neg = getTraining(neg_data)
training_data = training_pos + training_neg
#training_data.append(training_pos)
#training_data.append(training_neg)

testing_pos = getTesting(pos_data) 
testing_neg = getTesting(neg_data)
testing_data = testing_pos + testing_neg
#testing_data.append(testing_pos)
#testing_data.append(testing_neg)

# not needed
cv = CountVectorizer(decode_error ='ignore')
# not needed
tfidf_transformer = TfidfTransformer()
               		
text_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
            	 #	('clf', DecisionTreeClassifier()), 
            		('clf', SVC(kernel='rbf')),
        ])

# TAGGING the data - seperate array
# Note, a dictionary might be more appropriate - where the prompt maps to the tag (0/1) 
# Check if this is possible with sklearn 

training_pos_target = [0]*len(training_pos)
training_neg_target = [1]*len(training_neg)

pos_target = [0]*len(pos_data)
neg_target = [1]*len(neg_data)
target = pos_target + neg_target

## NOTE: Split this up into training and testing data; this is impossible to test
text_clf = text_clf.fit(pos_data + neg_data, target)
print text_clf

# my lame attempt at testing. Don't work
#docs_new = ['its so hard']
#newcount = cv.transform(docs_new)
#tf = tfidf_transformer.transform(newcount)
#predicted = text_clf.predict(tf)
#print predicted

#	CODE TO TEST: (http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
# >>> import numpy as np
# >>> twenty_test = fetch_20newsgroups(subset='test',
# ...     categories=categories, shuffle=True, random_state=42)
# >>> docs_test = twenty_test.data
# >>> predicted = text_clf.predict(docs_test)
# >>> np.mean(predicted == twenty_test.target)            
# 0.834...
# twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
# docs_test = twenty_test.data
# predicted = text_clf.predict(docs_test)
# print np.mean(predicted == twenty_test.target) 


testing_pos_target = [0]*len(testing_pos)	
testing_neg_target = [1]*len(testing_neg)

training_target = training_pos_target + training_neg_target
testing_target = testing_pos_target + testing_neg_target
## NOTE: Split this up into training and testing data; this is impossible to test
text_clf = text_clf.fit(training_data, training_target)

save(text_clf, 'svm_classifier.dump')
predicted = text_clf.predict(testing_data)

print np.mean(predicted == testing_target)
print (metrics.classification_report(testing_target, predicted))
print metrics.confusion_matrix(testing_target, predicted)




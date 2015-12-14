import pickle  
import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np


def load(filename):
	ifile = open(filename, 'r+')
	data = pickle.load(ifile)
	ifile.close()
	return data

# add a shuffle function somewhere here
# split into training and testing data 
all_data = load("data.dump")
alld = []
pos_data = all_data[0]
neg_data = all_data[1]
alld.append(pos_data)
alld.append(neg_data)

# not needed
cv = CountVectorizer(decode_error ='ignore')
# not needed
tfidf_transformer = TfidfTransformer()

text_clf = Pipeline([('vect', CountVectorizer(decode_error ='ignore', tokenizer=lambda doc: doc, lowercase=False)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                    ])

# TAGGING the data - seperate array
# Note, a dictionary might be more appropriate - where the prompt maps to the tag (0/1) 
# Check if this is possible with sklearn 
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









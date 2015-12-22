
Libraries used which may require a download: 
- sklearn (http://scikit-learn.org/stable/install.html)
- nltk (http://www.nltk.org/install.html)
- numpy (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- pickle (should not require an install) 
- random (should not require an install) 
- string (should not require an install)
- os (should not require an install)
- sys (should not require an install)


Imports: 
SVC, MultinomialNB, CountVectorizer, metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from nltk.stem.porter import *
from random import shuffle 

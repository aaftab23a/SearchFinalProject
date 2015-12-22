To run any file, simply navigate to the directory and run the command python filename.py. A function within the file can be called by editing the contents of the file. 

Specific Descriptions of the possible uses of each file:  
1. Run process_data.py, which reads data from the data folder, processes it and pickles it for later use.
2. Run naive_bayes.py, which will train a Naive Bayes classifier from the nltk library and pickle it for later use in nb_classifier.py. Note that naive_bayes.py is not used in the final model. It was used for initial testing before we moved to the SVM and Random_Forest classifiers. 
3. Run svm_classifier.py, which trains an SVM classifier on the processed data and tests it. It will print information to the console about the accuracy, precision and recall. Note that this does not only contain the SVM classifier. With minor changes to the train function, you can call the Multinomial Naive Bayes classifier and the Random Forest classifier as well. This is because we're using a Pipeline to process data and train the classifier. Sklearn does not require special processing for different types of classifiers, which makes it relatively simple to perform tests on a variety of classifiers.
4. Run Random_Forests.py, which will train an RandomForestClassifier on processed data and test it. 
5. ---- Run Stacked Learner?


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

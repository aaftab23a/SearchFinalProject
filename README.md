# Computational Humor Detection

## Goal 
This is an investigation into the effectiveness of text classification techniques in the art of detecting humor. 

## Background
Humor is a compelling and unique aspect of human behavior. While it has been studied extensively in linguistics and psychology, there has been very little work done in the field of computational humor - most of that work is on humor generation, not humor recognition, which is a subtle art. As part of our project, we wanted to find evidence that computational approaches can be successfully applied to the task of humor recognition. Since a deep understanding of humor in all of its aspects is an ambitious goal, we chose to restrict our investigation to the type of humor found in one-liners: specifically, that’s what she said jokes. Our experiment with data scraped from the web demonstrates that precision in the prediction job can be drastically improved with tailored text classification techniques.

## Techniques Used
We experimented with various types of classification methods, specifically the Naive Bayes classifiers, SVMs, Decision Trees and Random Forests. Our final model is a result of 'stacking' three of our most effective classifiers together (SVM trained on unigrams + SVM trained on parts of speech tags + Random Forest trained on unigrams). 
 

## How to run:  
To run any file, simply navigate to the directory and run the command 'python filename.py'. A function within the file can be called by editing the contents of the file. 

1. process_data.py: which reads data from the data folder, processes it and pickles it for later use. During processing, it portions the data into positive and negative, training and testing. These portions are picked and stored in the PickledData directory. 

2. naive_bayes.py: will train a Naive Bayes classifier from the nltk library and pickle it for later use in nb_classifier.py.  It will print information to the console about the accuracy, precision, and recall. Note that naive_bayes.py is not used in the final model. It was used for initial testing before we moved to the SVM and Random_Forest classifiers.

3. svm_classifier.py: trains an SVM classifier on the processed data and tests it. It will print information to the console about the accuracy, precision, and recall. Note that this does not only contain the SVM classifier. With minor changes to the train function, you can call the Multinomial Naive Bayes classifier and the Random Forest classifier as well. This is because we're using a Pipeline to process data and train the classifier. Sklearn does not require special processing for different types of classifiers, which makes it relatively simple to perform tests on a variety of classifiers.

4. Random_Forests.py: trains an RandomForestClassifier on processed data tests the classifier and prints information about accuracy, precision and recall to the console. 

5. stacked_learner.py: trains the SVM on POS tags carries out the stacked learner algorithm, and  prints out information about the accuracy, precision and recall to the console.

Here is a link to the data that we used, stored in google drive: https://drive.google.com/folderview?id=0B5omDrNLVLJEOXJiZ09HbC1MUzA&usp=sharing

##Libraries used which may require a download: 
- sklearn (http://scikit-learn.org/stable/install.html)
- nltk (http://www.nltk.org/install.html)
- numpy (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- treetaggerwrapper (https://pypi.python.org/pypi/treetaggerwrapper/2.0.6)
- pickle (should not require an install) 
- random (should not require an install) 
- string (should not require an install)
- os (should not require an install)
- sys (should not require an install)


### A note on the installation of the treetagger library:
  If you have trouble installing the library, just install 1 and use the source code folder at   https://github.com/NashwaEl/TreeTaggerInstall to install it: (add the folder to your working directory and run python treetaggerwrapper.py )
  1. http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/ (Treetagger library)
  2. http://treetaggerwrapper.readthedocs.org/en/latest/  (Treetagger library python wrapper)

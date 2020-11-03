## Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def Performance_Eval(twenty_train,twenty_test,text_clf,algo_perf_mean,algorithm):
    """
    Performance evaluation function for different algorithms. Takes input of the training and test dataset along with the pipeline
    to be run and returns the mean values along with the table as an output.
    """
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(twenty_test.data)
    means = np.mean(predicted == twenty_test.target)
    algo_perf_mean.append(means)
    
    ## Printing the performance analysis of the model 
    ## Can comment these two lines to prevent classification report from getting printed
    print('Classification report for %s: \n' % algorithm)
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
    
    print('Achieved Accuracy for %s : %.3f%s\n' % (algorithm,means*100,"%") )
    return(algo_perf_mean)

## Fetching the 20newsgroup dataset and splitting it into trainig and test data
trainset = fetch_20newsgroups(subset='train', shuffle=True, random_state=23)
testset = fetch_20newsgroups(subset='test', shuffle=True, random_state=23)

## Defining the classifier names which we will be using in this experiment
classifiers = ['Naïve Bayes (counts)','Naïve Bayes (tf)','Naïve Bayes (tf-idf)',
               'SVM (counts)', 'SVM (tf)', 'SVM (tf-idf)',
               'Random Forest (counts)','Random Forest (tf)','Random Forest (tf-idf)']
algos = np.array(["NB","SVM","RF"])
TotalM = []
## Designing a pipeline which sequentially apply a list of transforms and a final estimator.
for i in range(4):
    algo_perf_mean = []
    counter = 0
    ## Playing with count vectorizers
    ## CountVectorizer: Convert a collection of text documents to a matrix of token counts
    if i == 0:
        Vect = CountVectorizer(lowercase=True)
        Title = "LWRCase = TRUE"
    elif i == 1:
        Vect = CountVectorizer(lowercase=False)
        Title = "LWRCase = FALSE"
    elif i == 2:
        Vect = CountVectorizer(stop_words='english')
        Title = "stop_words = 'english'"
    elif i == 3:
        Vect = CountVectorizer(analyzer='char', ngram_range=(1, 4))
        Title = "analyzer with ngram_range"
    print(Title)
    for classifier in algos:
        if classifier == "NB":
            counts = MultinomialNB()
        elif classifier == "SVM":
            ## tol: is the stopping criteria; rest all parameters are kept as default
            ## max_iter: is the maximum number of passes over the training data 
            counts = SGDClassifier(random_state=23, max_iter=10, tol=None) 
        elif classifier == "RF":
            ## n_estimator is the number of trees in the forest.
            counts = RandomForestClassifier(n_estimators=100)
        ## Counts
        text_clf = Pipeline([('vect', Vect),('clf', counts)])
        algo_perf_mean = Performance_Eval(trainset,testset,text_clf,algo_perf_mean,classifiers[counter*3])
        ## Term Frequencies (tf)
        text_clf = Pipeline([('vect', Vect),('tfidf', TfidfTransformer(use_idf=False)),('clf', counts)])
        algo_perf_mean = Performance_Eval(trainset,testset,text_clf,algo_perf_mean,classifiers[((counter*3)+1)])
        ## Term frequency times inverse document frequency (tf-idf)
        text_clf = Pipeline([('vect', Vect),('tfidf', TfidfTransformer()),('clf', counts)])
        algo_perf_mean = Performance_Eval(trainset,testset,text_clf,algo_perf_mean,classifiers[((counter*3)+2)])
        counter+= 1
    TotalM.append(algo_perf_mean)
    print('\n\nAlgorithm with highest accuracy for %s is %s: %.3f%s\n' % (Title,str(classifiers[algo_perf_mean.index(max(algo_perf_mean))]), (max(algo_perf_mean)*100) , '%'))
    Analysis = pd.DataFrame(np.array(algo_perf_mean).reshape(len(algos),3),columns=["counts","tf","tf-idf"],index=algos).T
    if i == 0:
        plt.subplot(221)    
    elif i == 1:
        plt.subplot(222)    
    elif i == 2:
        plt.subplot(223)    
    elif i == 3:
        plt.subplot(224)    
    plt.plot(Analysis*100, '--', linewidth = 1.5, marker = ".", ms = 10)
    plt.tight_layout()
    plt.ylabel("Percentage Accuracy")
    plt.title(Title)
    plt.ylim((0,100))
    plt.xlabel("Features")
    plt.legend(algos)
plt.show()
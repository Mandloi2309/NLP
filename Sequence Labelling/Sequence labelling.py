import numpy as np
import sklearn_crfsuite
from os import listdir
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.model_selection import KFold
from string import punctuation
from re import search
from pandas import DataFrame
import matplotlib.pyplot as plt

## Libraries for Hyperparameter optimization:
#import scipy.stats
#from sklearn.metrics import make_scorer
#from sklearn.model_selection import RandomizedSearchCV

def getData(directory):
    '''
    This function takes in a directory path which has the input file.
    It reads .bio the files and creates a list of tuple lists.
    returns the preprocessed list.
    
    '''
    files  = listdir(directory) 
    Bio = []
    FileCounter = 0
    index = []
    for file in files:
        BioData = []
        fh = open(str(directory+file),'r', encoding='utf8')
        for i,line in enumerate(fh):
            vecs = str.split(line.replace('\n',''),'\t')
            if len(vecs)==1:
                index.append([FileCounter,file,i])
            BioData.append(tuple(vecs))
        fh.close()
        Bio.append(BioData)
        FileCounter+=1
    return(preProcessingIOB(Bio,index))


def preProcessingIOB(Bio,index):
    '''
    Since data may have some inconsistency so this preprocessing funtion is used.
    Wherever the function finds any inconsistencies in the data it assigns the POS tag
    and the IOB tag of the previous entry to it.
    
    '''
    if len(index)>0:
        for i,j in enumerate(index):
            doc = j[0]
            pos = j[2]
            Bio[doc][pos] = tuple([Bio[doc][pos][0], Bio[doc][pos-1][1], Bio[doc][pos-1][2]])
    return(Bio)

def isspecial(word):
    if search('['+punctuation+']', word):
        return(True)
    else:
        return(False)

### Converting each word into features    
def word2features(sent, i, mode):
    '''
    Converts each word into an object of features, also takes an input mode which decides whether to use 
    default features or custom features.
    
    Returns an object with the features
    
    '''
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),                       # Normalizing
        'word[-3:]': word[-3:],                             # Suffix
        'word[:2]': word[:2],                               # Prefix
        'word.isupper()': word.isupper(),                   # Uppercase
        'word.istitle()': word.istitle(),                   # Title
        'word.isdigit()': word.isdigit(),                   # Numeric
        'postag': postag,                                   # POS tag
        'postag[:2]': postag[:2],
    }
    if mode == 'C':
        ## Custom features
        features.update({
                'word.isonlycharacters()': word.isalpha(),  # Only character
                'word.isalphanumeric()': word.isalnum(),    # Alphanumeric
                'word.isspecialchar()': isspecial(word)     # Special Characters
        })
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOD'] = True                              # Begining of a document

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOD'] = True                              # End of a document

    return features


def sent2features(sent, mode = 'D'):
    '''
    Calls word2features for each word
    '''
    return [word2features(sent, i, mode) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def CRFModel(crf, pat_train, pat_test, mode):
    '''
    Input parameters: Train and test dataset, mode to decide whether to use 
    default features or custome features
    Peforms sequence labelling using crfsuite, and calculates the f1 score for the 
    predicted values.
    
    returns the crf model along with the F1 score.
    '''
    
    
    ## Train set 
    X_train = [sent2features(s, mode) for s in pat_train]
    y_train = [sent2labels(s) for s in pat_train]
    ## Test set
    X_test = [sent2features(s, mode) for s in pat_test]
    y_test = [sent2labels(s) for s in pat_test]
    
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)
    F1 = metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)
    print('Balanced F-score: %.4f' %F1)

    # group B and I results
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    return(crf, F1)
    
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

def print_report(crf):
    print("Top 5 likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(5))
    print("\nTop 5 unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-5:])
    print("Top 10 positive:")
    print_state_features(Counter(crf.state_features_).most_common(10))
    print("\nTop 10 negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-10:])

    
patent_citation = getData('bio/')
kf = KFold(n_splits=5,shuffle=True, random_state=23)
# Define fixed parameters and parameters to search
feature_mode = ['D', 'C']
opt_mode = ['N', 'Y']
Final_F1 = []
for mode in feature_mode:
    if mode == 'D':
        print('\nUsing default features...\n')
    elif mode == 'C':
        print('\nUsing custom features...\n')
    for op in opt_mode:
        if op == 'Y':
            crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs',max_iterations = 100, 
                    c1 = 0.161816, c2 = 0.054251, ## Optimization parameters
                    all_possible_transitions = True)
            print('Running CRFsuite with the optimization parameters...\n')
        elif op == 'N':
            crf = sklearn_crfsuite.CRF(algorithm = 'lbfgs',max_iterations = 100, 
                    all_possible_transitions = True)
            print('Running CRFsuite without the optimization parameters...\n')
        
        F1_score = []
        for train, test in kf.split(patent_citation[0:18]):
            crf,F1 = CRFModel(crf, np.array(patent_citation)[train].tolist(), np.array(patent_citation)[test].tolist(), mode)
            F1_score.append(F1) 
            
        ## Prediction on the test set:
        labels = list(crf.classes_)
        labels.remove('O')
        X_test = [sent2features(s) for s in patent_citation[18:23]]
        y_test = [sent2labels(s) for s in patent_citation[18:23]]
        y_pred = crf.predict(X_test)
        mean_F1 = sum(F1_score)/len(F1_score)
        F1_test = metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)
        print('Mean F1 score: %.4f, Test F1 score: %.4f\n' %(mean_F1,F1_test))
        print_report(crf)
        Final_F1.append([mode, op, mean_F1, F1_test])

Final = DataFrame(Final_F1, columns=['Feature', 'Optimized', 'Mean F1', 'Test F1'])
Final.plot(kind = 'bar')
plt.xlabel('CRF feature settings')
plt.ylabel('F1 Score')
plt.ylim((0,1.2))
plt.title('F1 scores of CRF model')
plt.xticks(ticks = [0,1,2,3], 
           labels = ['Default\n(Not opt)', 'Default\n(Opt)', 'Custom\n(Not opt)', 'Custom\n(Opt)'])
plt.show()
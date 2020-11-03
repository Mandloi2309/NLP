from os import listdir, getcwd
from os.path import isfile, join
from re import sub, compile as comp
from string import punctuation
from time import time
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  LinearSVC
start = time()

import logging
import sys
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -- %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def LookFor(mypath, check):
    return([f for f in listdir(mypath) if isfile(join(mypath, f)) != check])

def ReadFiles(filepath, files):
    ## Reads the individual files from the provided path and pre-processes the data
    stpwd = ""
    for i, j in enumerate(stopwords):
        stpwd = stpwd+j
        stpwd = stpwd+"|"        
    stpwd = stpwd[0:(len(stpwd)-1)]
    outdata = []
    for x in files:
      reader = open(join(filepath, x), "r", encoding = enco)
      ## Preprocessing the data
      outstring = reader.readline().lower().replace("<br />", " ")  ## Remove html tags and lowercase data
      outstring = sub("\s["+stpwd+"]\s", " ", outstring)            ## Remove stop-words
      outstring = sub("["+punctuation+"]","",outstring)             ## Remove punctuations
      outdata.append(sub("\s{2,}"," ",outstring))                   ## Remove double spaces
      reader.close()
    return(outdata)

def numericalSort(value):
    ## Numerically orders files
    numbers = comp(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def Filecreating(mypath):
    ## Merges all individual review after pre-processing the data
    folders = LookFor(mypath, True)
    subfolder = ["pos", "neg"]
    for fldr in folders:
        print("Reading %s dataset" %fldr.upper())
        for cat in subfolder:
            filepath = join(mypath, fldr, cat)
            print("\t%s reviews..." % cat.upper())
            data = ReadFiles(filepath, sorted(LookFor(filepath, False), key=numericalSort))
            outputfile = open(fldr.upper()+"_"+cat.upper()+".txt", "w", encoding=enco)
            outputfile.close()
            outputfile = open(fldr.upper()+"_"+cat.upper()+".txt", "a", encoding=enco)
            for i in range(len(data)):
                outputfile.write(data[i])
                outputfile.write("\n")
            outputfile.close()

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.open(source, encoding=enco) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.open(source, encoding=enco) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)


def createTrainTest(model, size, test_sentences):
    train_arrays = numpy.zeros((25000, size))
    train_labels = numpy.zeros(25000)
    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[12500 + i] = 0
    test_arrays = numpy.zeros((25000, size))
    test_labels = numpy.zeros(25000)
    for index, i in enumerate(test_sentences):
        feature = model.infer_vector(i[0])
        if index <12500:
            test_arrays[index] = feature
            test_labels[index] = 0
        else:
            test_arrays[index] = feature
            test_labels[index] = 1

    return(train_arrays, train_labels, test_arrays, test_labels)

def WordEmbedding(string, min_count, size, window, negs, epochs):
    train_source = {'TRAIN_NEG.txt':'TRAIN_NEG', 'TRAIN_POS.txt':'TRAIN_POS'}
    test_source = {'TEST_NEG.txt':'TEST_NEG', 'TEST_POS.txt':'TEST_POS'}
    #log.info('TaggedDocument')
    train_sentences = TaggedLineSentence(train_source)
    test_sentences = TaggedLineSentence(test_source)
    print(string)
    model = Doc2Vec(min_count=min_count, window=window, vector_size=size, sample=1e-4, negative=negs, workers=5,epochs=epochs)
    model.build_vocab(train_sentences.to_array())
    # log.info('EPOCH: {}'.format(model.iter))
    model.train(train_sentences.sentences_perm(), total_examples=model.corpus_count, epochs=model.epochs)
    model.save('./imdb_'+string+'.d2v')
    model = Doc2Vec.load('./imdb_'+string+'.d2v')
    print(model.wv.most_similar('must'))
    print(model.wv.most_similar('amazing'))
    print(model.wv.most_similar('horrible'))
    
    ## Checking the most similar documents to 1st document
    max_sim = 0
    same_file  = ""
    for i in range(2, 12500):
        X = model.docvecs.similarity("TRAIN_POS_1", "TRAIN_POS_"+str(i))
        if(X>max_sim):
            max_sim = X
            same_file = "TRAIN_POS_"+str(i)
            
    print("File with maximum similarity with the TRAIN_POS_1 file is %s with similarity of %.3f" %(same_file, max_sim))
      
    train_arrays, train_labels, test_arrays, test_labels = createTrainTest(model, size, test_sentences)
    ## Classifier Linear SVC
    classifier = LinearSVC()
    classifier.fit(train_arrays, train_labels)
    print("Linear SVC "+string+": %.4f" % classifier.score(test_arrays, test_labels))

    ## CLassifier Logistic Regression
    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    classifier.fit(train_arrays, train_labels)
    print("Logistic Regression "+string+": %.4f" %classifier.score(test_arrays, test_labels))
        

    
enco = "utf8"
mypath = join(getcwd(), "aclImdb")
Filecreating(mypath)

WordEmbedding(string="STP_Model_1", min_count=1, size=150, window=10, negs=5, epochs=50)
ETA1 = time()
print("Time Elapsed: %.3f secs" % (ETA1 - start))

print("Total Time Elapsed: %.3f secs" % (time() - start))
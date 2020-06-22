# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:02:24 2020

@author: Maya
HW4 - Deception and Subjectivity
"""

import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer

#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer but with tf-idf norm
from sklearn.feature_extraction.text import TfidfVectorizer


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import os.path

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import scikitplot as skplt

# Read in the data 
df = pd.read_csv('deception_data_converted_final.csv',  sep='\t')

## Check the head 
print(df.head(5))


# Get the labels, they are at the beggining
def get_labels(row):
    split_row = str(row).split(',') 
    lie = split_row[0] # split t and f
    sentiment = split_row[1] # split p and n
    return [lie, sentiment, split_row[2:]] 

df['new_all'] = df.apply(lambda row: get_labels(row['lie,sentiment,review']), axis=1) # create new column
print(df.head(5))

# Add Lie column
df['lie'] = df.apply(lambda row: row['new_all'][0][0], axis=1)
#print(df.head(5))

# Add sentiment column
df['sentiment'] = df.apply(lambda row: row['new_all'][1][0], axis=1)
#print(df.head(5))

# Add review column
df['review'] = df.apply(lambda row: ''.join(row['new_all'][2]), axis=1)
#print(df.head(5))

# Make a copy
clean_df = df.copy()

# Leave only the want I need 
clean_df.drop(['lie,sentiment,review', 'new_all'], axis=1, inplace=True)

print(clean_df)

# Now we have 3 columns, have to clean the reviews a little 

def clean_rogue_characters(string):
    exclude = ['\\',"\'",'"']
    string = ''.join(string.split('\\n'))
    string = ''.join(ch for ch in string if ch not in exclude)
    return string

clean_df['review'] = clean_df['review'].apply( lambda x: clean_rogue_characters(x) )
print(clean_df['review'][0])

#print(clean_df)

# Save to csv file 
clean_df.to_csv('deception_new.csv',index=False) # optional or just work with the clean_df

df = pd.read_csv('deception_new.csv')
print(df.head())

# average number of words per sample
plt.figure(figsize=(10, 6))
plt.hist([len(sample) for sample in list(df['review'])], 49)
plt.xlabel('Length of samples')
plt.ylabel('Number of samples')
plt.title('Sample length distribution')
plt.show()

#############################################################################
# Split it into 2 lie datasets
lie_f = df[df['lie'] == 'f']
lie_t = df[df['lie'] == 't']

# Split into 2 sentiment datsets
sent_n = df[df['sentiment'] == 'n']
sent_p = df[df['sentiment'] == 'p']


# Export 
def print_to_file(rating, review, num, title):
    both = review
    output_filename = str(rating) + '_'+ title +'_' + str(num) + '.txt'
    outfile = open(output_filename, 'w')
    outfile.write(both)
    outfile.close()

def export_to_corpus(df, subj, title):
    for num,row in enumerate(df['review']):
        print_to_file(subj, row, num, title)
        
#export_to_corpus(sent_n, 'neg', 'sen_n')
#export_to_corpus(sent_p, 'pos', 'sen_p')

#export_to_corpus(lie_f, 'false', 'lie_f')
#export_to_corpus(lie_t, 'true', 'lie_t')

############################    Second Way ##################################
print(df.head())

# Make a copy
lie_df = df.copy()

# create df only for lie and review 
lie_df.drop(['sentiment'], axis=1, inplace=True)
print(lie_df.head())
# Write to csv
lie_df.to_csv('lie_reviews.csv',index=False)


# Make a copy
sentiment_df = df.copy()

# create df only for lie and review 
sentiment_df.drop(['lie'], axis=1, inplace=True)
print(sentiment_df.head())
# Write to csv
sentiment_df.to_csv('sentiment_reviews.csv',index=False)

#############################################################################

RawfileName="C:/Users/aivii/programsmm/HW4_736/lie_reviews.csv"
RawfileName0="C:/Users/aivii/programsmm/HW4_736/sentiment_reviews.csv"

## This file has a header. 
## It has "lie" and "review" on the first row.


# FIRST DF - LIE AND REVIEWS
AllReviewsList=[]
AllLabelsList=[]

with open(RawfileName,'r') as FILE:
    FILE.readline() # skip header line - skip row 1
    ## This reads the line and so does nothing with it
    for row in FILE:
        NextLabel,NextReview=row.split(",", 1)
        #print(Label)
        #print(Review)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel)  
    
print(AllReviewsList)
print(AllLabelsList)

My_CV1=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True
                        
                        )

# Now I can vectorize using my list of complete paths to my files
X_CV1=My_CV1.fit_transform(AllReviewsList)

print(My_CV1.vocabulary_)

ColNames=My_CV1.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CV = DataFrame_CV.rename(MyDict, axis = "index")
DataFrame_CV.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CV.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_CV.columns:
    #print(nextcol)
    ## Remove unwanted columns
    Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        

DataFrame_CV1 = DataFrame_CV.reset_index()
#print(DataFrame_CV)
lie_df = DataFrame_CV1.copy()
print(lie_df) # for models 


# SECOND DF - SENTIMENT AND REVIEWS
AllReviewsList=[]
AllLabelsList=[]

with open(RawfileName0,'r') as FILE:
    FILE.readline() # skip header line - skip row 1
    ## This reads the line and so does nothing with it
    for row in FILE:
        NextLabel,NextReview=row.split(",", 1)
        #print(Label)
        #print(Review)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel)  
    
print(AllReviewsList)
print(AllLabelsList)

My_CV2=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True
                        
                        )

# Now I can vectorize using my list of complete paths to my files
X_CV2=My_CV2.fit_transform(AllReviewsList)

print(My_CV2.vocabulary_)

ColNames=My_CV2.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV2=pd.DataFrame(X_CV2.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CV2 = DataFrame_CV2.rename(MyDict, axis = "index")
DataFrame_CV2.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CV2.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_CV2.columns:
    #print(nextcol)
    ## Remove unwanted columns
    Result=str.isdigit(nextcol) ## Fast way to check numbers
    print(Result)
    
    
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV2=DataFrame_CV2.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV2=DataFrame_CV2.drop([nextcol], axis=1)
        

DataFrame_CV3 = DataFrame_CV2.reset_index()
#print(DataFrame_CV)
sentiment_df = DataFrame_CV3.copy()
print(sentiment_df) # for models 

# another way was just to work for thr reviews and just add on the labels after

#print(lie_df)
#print(sentiment_df)

#######################################################################

# add Bernoli 
## This will be used to vectorize the text data for Bern
### Notice that this one is binary - which is used for Bernoulli
MyVectCV1B=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=LemmaTokenizer(),
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        binary=True

                        )


#######################################################################
# Train test split -  LIE DF
## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split

TrainLie, TestLie = train_test_split(lie_df, test_size=0.3)

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainLie)
print("\nThe testing set is:")
print(TestLie)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabelsLie=TestLie["LABEL"]
#print(TestLabels)
## remove labels
## Make a copy of TestLie
CopyTestDFLie=TestLie.copy()
TestLie = TestLie.drop(["LABEL"], axis=1)
print(TestLie)

## DF seperate TRAIN SET from the labels
TrainLie_nolabels=TrainLie.drop(["LABEL"], axis=1)
#print(TrainDF_nolabels)
TrainLabelsLie=TrainLie["LABEL"]
#print(TrainLabels)

##############################################################
# Train test split -  SENTIMENT DF

TrainSent, TestSent = train_test_split(sentiment_df, test_size=0.3)

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainSent)
print("\nThe testing set is:")
print(TestSent)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabelsSent=TestSent["LABEL"]
#print(TestLabels)
## remove labels
## Make a copy of TestLie
CopyTestDFSent=TestSent.copy()
TestSent = TestSent.drop(["LABEL"], axis=1)
print(TestSent)

## DF seperate TRAIN SET from the labels
TrainSent_nolabels=TrainSent.drop(["LABEL"], axis=1)
#print(TrainSent_nolabels)
TrainLabelsSent=TrainSent["LABEL"]
#print(TrainLabels)

#############################################################################
### MODELS
#############################################################################

### Naive Bayes

## LIE DF

from sklearn.naive_bayes import MultinomialNB

MyModelNB0= MultinomialNB()


MyModelNB0.fit(TrainLie_nolabels, TrainLabelsLie)
Prediction = MyModelNB0.predict(TestLie)

print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabelsLie)

## confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(TestLabelsLie, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
#plt.imshow(cnf_matrix, cmap ='binary')

skplt.metrics.plot_confusion_matrix(TestLabelsLie, Prediction,normalize=False,figsize=(12,8))
plt.show()

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB0.predict_proba(TestLie),2))

from sklearn import metrics

print(metrics.classification_report(TestLabelsLie, Prediction))
print(metrics.confusion_matrix(TestLabelsLie, Prediction))

def rev_important_features(vectorizer,classifier, n):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.feature_count_[0], feature_names), reverse = True)[:n]
    topn_class2 = sorted(zip(classifier.feature_count_[1], feature_names), reverse = True)[:n]
    print("Important words in negative reviews: ")
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
    print("-----------------------------------------")
    print("Important words in positive reviews: ")
    for coef, feat in topn_class2:
        print(class_labels[1], coef, feat)

print("Try this:")        
rev_important_features(My_CV1,MyModelNB0, 20)


# another way 

# top positive and negative features 
feature_to_coef = {
    word: coef for word, coef in zip(
        My_CV1.get_feature_names(), MyModelNB0.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:20]:
    print (best_positive)
    
print("\n\n")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:20]:
    print (best_negative)



## SENTIMENT DF

MyModelNB= MultinomialNB()


MyModelNB.fit(TrainSent_nolabels, TrainLabelsSent)
Prediction = MyModelNB.predict(TestSent)

print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabelsSent)

## confusion matrix
cnf_matrix = confusion_matrix(TestLabelsSent, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
#plt.imshow(cnf_matrix, cmap ='binary')

skplt.metrics.plot_confusion_matrix(TestLabelsSent, Prediction,normalize=False,figsize=(12,8))
plt.show()

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB.predict_proba(TestSent),2))

print(metrics.classification_report(TestLabelsSent, Prediction))
print(metrics.confusion_matrix(TestLabelsSent, Prediction))


# top positive and negative features 
feature_to_coef = {
    word: coef for word, coef in zip(
        My_CV2.get_feature_names(), MyModelNB.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:20]:
    print (best_positive)
    
print("\n\n")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:20]:
    print (best_negative)




#############################################################################
### Bernoulli 

## LIE DF
from sklearn.naive_bayes import BernoulliNB

## Bernoulli uses 0 and 1 data (not counts)
## So - we need to re-format our data first
## Make a COPY of the DF
TrainLie_nolabels_Binary=TrainLie_nolabels.copy()   ## USE .copy()
TrainLie_nolabels_Binary[TrainLie_nolabels_Binary >= 1] = 1
TrainLie_nolabels_Binary[TrainLie_nolabels_Binary < 1] = 0

print(TrainLie_nolabels_Binary)

BernModel = BernoulliNB()
BernModel.fit(TrainLie_nolabels_Binary, TrainLabelsLie)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PredictionB = BernModel.predict(TestLie)
print("\nBernoulli prediction:\n", BernModel.predict(TestLie))
print("\nActual:")
print(TestLabelsLie)

bn_matrix = confusion_matrix(TestLabelsLie, BernModel.predict(TestLie))
print("\nThe confusion  Blie matrix is:")
print(bn_matrix)

skplt.metrics.plot_confusion_matrix(TestLabelsLie, PredictionB,normalize=False,figsize=(12,8))
plt.show()

print(np.round(BernModel.predict_proba(TestLie),2))

print(metrics.classification_report(TestLabelsLie, PredictionB))
print(metrics.confusion_matrix(TestLabelsLie, PredictionB))

## SENTIMENT DF
from sklearn.naive_bayes import BernoulliNB

## Bernoulli uses 0 and 1 data (not counts)
## So - we need to re-format our data first
## Make a COPY of the DF
TrainSent_nolabels_Binary=TrainSent_nolabels.copy()   ## USE .copy()
TrainSent_nolabels_Binary[TrainSent_nolabels_Binary >= 1] = 1
TrainSent_nolabels_Binary[TrainSent_nolabels_Binary < 1] = 0

print(TrainSent_nolabels_Binary)

BernModel1 = BernoulliNB()
BernModel1.fit(TrainSent_nolabels_Binary, TrainLabelsSent)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("\nBernoulli prediction:\n", BernModel1.predict(TestSent))
print("\nActual:")
print(TestLabelsSent)

PredictionB1 = BernModel1.predict(TestSent)
bn_matrix1 = confusion_matrix(TestLabelsSent, PredictionB1)
print("\nThe confusion Bsent matrix is:")
print(bn_matrix1)

skplt.metrics.plot_confusion_matrix(TestLabelsSent, PredictionB1 ,normalize=False,figsize=(12,8))
plt.show()

print(np.round(BernModel1.predict_proba(TestSent),2))

print(metrics.classification_report(TestLabelsSent, PredictionB1))
print(metrics.confusion_matrix(TestLabelsSent, PredictionB1))

#############################################################################
## SVM 

## LIE DF
from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=.01)
SVM_Model.fit(TrainLie_nolabels, TrainLabelsLie)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestLie))
print("Actual:")
print(TestLabelsLie)

SVM_matrix = confusion_matrix(TestLabelsLie, SVM_Model.predict(TestLie))
print("\nThe confusion  SVMlie matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TestLabelsLie, SVM_Model.predict(TestLie) ,normalize=False,figsize=(12,8), cmap=plt.cm.Oranges_r)
plt.show()

print(metrics.classification_report(TestLabelsLie, SVM_Model.predict(TestLie)))
print(metrics.confusion_matrix(TestLabelsLie, SVM_Model.predict(TestLie)))

cmap=plt.cm.Blues

## SENTIMENT DF

SVM_Model1=LinearSVC(C=.01)
SVM_Model1.fit(TrainSent_nolabels, TrainLabelsSent)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model1.predict(TestSent))
print("Actual:")
print(TestLabelsSent)

SVM_matrix = confusion_matrix(TestLabelsSent, SVM_Model1.predict(TestSent))
print("\nThe confusion SVMsentiment matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TestLabelsSent, SVM_Model1.predict(TestSent) ,normalize=False,figsize=(12,8), cmap=plt.cm.Oranges_r)
plt.show()

print(metrics.classification_report(TestLabelsSent, SVM_Model1.predict(TestSent)))
print(metrics.confusion_matrix(TestLabelsSent, SVM_Model1.predict(TestSent)))

##############################################################################

# wordcloulds - text 
# combine the text for word cloulds 

# lie - fasle
path = "C:\\Users\\aivii\\OneDrive\Desktop\\HW4 736\\false_lie"
#print(os.listdir(path))
false = os.listdir(path) # now I have list of filenames 

false_alltext = []
for file in false:
    f=open('false_lie/'+file)
    content=f.read()
    false_alltext.append(content)
    f.close()

#print(false_alltext)

# lie - tr 
path = "C:\\Users\\aivii\\OneDrive\Desktop\\HW4 736\\true_lie"
#print(os.listdir(path))
true = os.listdir(path) # now I have list of filenames 

true_alltext = []
for file in true:
    f=open('true_lie/'+file)
    content=f.read()
    true_alltext.append(content)
    f.close()

# print(true_alltext)

# sent- neg 
path = "C:\\Users\\aivii\\OneDrive\Desktop\\HW4 736\\neg_sent"
#print(os.listdir(path))
neg = os.listdir(path) # now I have list of filenames 

neg_alltext = []
for file in neg:
    f=open('neg_sent/'+file)
    content=f.read()
    neg_alltext.append(content)
    f.close()

#print(neg_alltext)

# sent- pos 
path = "C:\\Users\\aivii\\OneDrive\Desktop\\HW4 736\\pos_sent"
#print(os.listdir(path))
pos = os.listdir(path) # now I have list of filenames 

pos_alltext = []
for file in pos:
    f=open('pos_sent/'+file)
    content=f.read()
    pos_alltext.append(content)
    f.close()

#print(pos_alltext)

#############################################################################
# top feat
print("Try this:")        
rev_important_features(My_CV1,MyModelNB0, 20)
rev_important_features(My_CV2,MyModelNB, 20)
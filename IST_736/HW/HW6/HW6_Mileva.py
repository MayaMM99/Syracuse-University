# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:41:03 2020

@author: Maya
HW4 - Deception and Subjectivity / Benoulli and Multinomian Naive Bayes in Sci-kit Learn
"""

## PART 1 - code with the small data

# 1. Count Vect
# 2. Count Vect for B(binary=True)
# 3. TFIdf

## Textmining Naive Bayes Example
import nltk
import pandas as pd
import sklearn
import re  
#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer but with tf-idf norm
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
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
import scikitplot as skplt
from sklearn import metrics

## This one will be used to vectorize the text data for NB
MyVect=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True
                        )

## This will be used to vectorize the text data for Bern
### Notice that this one is binary - which is used for Bernoulli
MyVectB=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True,
                        binary=True
                        )

MyVectTF=TfidfVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True
                        )

# We will be creating new data frames - one for NB and one for Bern and one for tfidf
## These are the three new and currently empty DFs
FinalDF=pd.DataFrame()
FinalDFB=pd.DataFrame()
FinalDFidf=pd.DataFrame()


## Loop through the files in these two folders and will build the list needed to use
## CounterVectorizer and tfidf

for name in ["DOG", "TRAVEL"]:

    builder=name+"DF"
    #print(builder)
    builderB=name+"DFB"
    builderT=name+"DFidf"
    path="C:\\Users\\aivii\\programsmm\\HW6_736\\" + name
    
    FileList=[]
    for item in os.listdir(path):
        #print(path+ "\\" + item)
        next=path+ "\\" + item
        FileList.append(next)  
        #print("full list...")
        #print(FileList)
        
        X=MyVect.fit_transform(FileList)
        XB=MyVectB.fit_transform(FileList)
        XTF= MyVectTF.fit_transform(FileList)
        ColumnNames2=MyVect.get_feature_names()
        #print("Column names: ", ColumnNames2)
        #Create a name
        
    #print(builder)
    builder=pd.DataFrame(X.toarray(),columns=ColumnNames2)
    builderB=pd.DataFrame(XB.toarray(),columns=ColumnNames2)
    builderT=pd.DataFrame(XTF.toarray(),columns=ColumnNames2)
    ## Add column
    print("Adding new column....")
    builder["Label"]=name
    builderB["Label"]=name
    builderT["Label"]=name
    print(builder)
    
    FinalDF = FinalDF.append(builder)
    FinalDFB = FinalDFB.append(builderB)
    FinalDFidf = FinalDFidf.append(builderT)
    #print(FinalDF)

#print(FinalDF)
#print(FinalDFB)
#print(FinalDFidf)

## Replace the NaN with 0 because it actually 
## means none in this case
FinalDF=FinalDF.fillna(0)
FinalDFB=FinalDFB.fillna(0)
FinalDFidf=FinalDFidf.fillna(0)


print("FIRST...Normal DF Freq")  ## These print statements help you to see where you are
print(FinalDF)

print("\nBINARY DF....")
print(FinalDFB)

print("\nTFIDF DF....")
print(FinalDFidf)

# Save 
#FinalDFidf.to_csv(r'C:\Users/aivii\programsmm\HW6_736\tf1.csv', index = False)
#FinalDF.to_csv(r'C:\Users/aivii\programsmm\HW6_736\cv.csv', index = False)
#FinalDFB.to_csv(r'C:\Users/aivii\programsmm\HW6_736\ber.csv', index = False)


## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split

######## Count Vect ##############################

## Create Train/Test for TrainDF
TrainDF, TestDF = train_test_split(FinalDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["Label"]
print(TestLabels)
## remove labels
TestDF = TestDF.drop(["Label"], axis=1)
print(TestDF)

TrainDF_nolabels=TrainDF.drop(["Label"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["Label"]
print(TrainLabels)


######## Count Vect Binary ######################

## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(FinalDFB, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["Label"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["Label"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["Label"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["Label"]
print(TestLabelsB)

######## TFIDF ######################

## Create Train/Test for TrainDFidf
TrainDFidf, TestDFidf = train_test_split(FinalDFidf, test_size=0.3)
print(TrainDFidf.head())

TrainDFidf_nolabels=TrainDFidf.drop(["Label"], axis=1)
print(TrainDFidf_nolabels)
TrainLabelsidf=TrainDFidf["Label"]
print(TrainLabelsidf)

TestDFidf_nolabels=TestDFidf.drop(["Label"], axis=1)
print(TestDFidf_nolabels)
TestLabelsidf=TestDFidf["Label"]
print(TestLabelsidf)


####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB


#### Model 1: Count Vect 
#Create the modeler
MyModelNB= MultinomialNB()


MyModelNB.fit(TrainDF_nolabels, TrainLabels)
print(MyModelNB.class_log_prior_) ## The greater the neg value - the more prob
## So -.2 has a greater probability than -1.2 (which is smaller than -.2)
Prediction = MyModelNB.predict(TestDF)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabels)

## confusion matrix
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)

print(np.round(MyModelNB.predict_proba(TestDF),2))

skplt.metrics.plot_confusion_matrix(TestLabels, Prediction ,normalize=False,figsize=(12,8), cmap=plt.cm.RdGy)
plt.show()

print(np.round(MyModelNB.predict_proba(TestDF),2))

print(metrics.classification_report(TestLabels, Prediction))
print(metrics.confusion_matrix(TestLabels, Prediction))


## Rankings

print("FOR TRAVEL:")
FeatureRanks=sorted(zip(MyModelNB.coef_[0],MyVect.get_feature_names()))
Ranks1=FeatureRanks[-20:]
## Make them unique
Ranks1 = list(set(Ranks1))
print(Ranks1)

print("\n\nFOR DOGS:")

FeatureRanks=sorted(zip(MyModelNB.coef_[0],MyVect.get_feature_names()),reverse=True)
Ranks2=FeatureRanks[-20:]
## Make them unique
Ranks2 = list(set(Ranks2))
print(Ranks2)

# Accuracies - another way
#MyModelNB.score(TestDF,TestLabels)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
#print(precision_score(TestLabels,Prediction, pos_label='DOG'))
#print(recall_score(TestLabels.tolist(),Prediction, pos_label='DOG'))
#print(classification_report(TestLabels,Prediction, target_names=['DOG','HIKE']))


#### Model 2: Count Vect - Bernoulli
# This should use the Binary

from sklearn.naive_bayes import BernoulliNB
BernModel = BernoulliNB()
BernModel.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("\nBernoulli prediction:\n", 
      BernModel.predict(TestDFB_nolabels))
print("\nActual:")
print(TestLabelsB)

bn_matrix = confusion_matrix(TestLabelsB, BernModel.predict(TestDFB_nolabels))
print("\nThe confusion matrix is:")
print(bn_matrix)

skplt.metrics.plot_confusion_matrix(TestLabelsB, BernModel.predict(TestDFB_nolabels) ,normalize=False,figsize=(12,8), cmap=plt.cm.RdGy)
plt.show()

#print(np.round(TestLabelsB, BernModel.predict(TestDFB_nolabels)),2)

print(metrics.classification_report(TestLabelsB, BernModel.predict(TestDFB_nolabels)))
print(metrics.confusion_matrix(TestLabelsB, BernModel.predict(TestDFB_nolabels)))


## Rankings

print("FOR TRAVEL:")
FeatureRanks=sorted(zip(BernModel.coef_[0],MyVectB.get_feature_names()))
Ranks2=FeatureRanks[-20:]
## Make them unique
Ranks2 = list(set(Ranks1))
print(Ranks2)

print("\n\nFOR DOGS:")

FeatureRanks=sorted(zip(BernModel.coef_[0],MyVectB.get_feature_names()),reverse=True)
Ranks3=FeatureRanks[-20:]
## Make them unique
Ranks2 = list(set(Ranks3))
print(Ranks3)


#### Model 3: TFIDF
## Create Train/Test for TrainDFidf
MyModelNB1= MultinomialNB()
MyModelNB1.fit(TrainDFidf_nolabels, TrainLabelsidf)
print(MyModelNB1.class_log_prior_) ## The greater the neg value - the more prob
## So -.2 has a greater probability than -1.2 (which is smaller than -.2)

print("\nTFIDF prediction:\n", 
      MyModelNB1.predict(TestDFidf_nolabels))
print("\nActual:")
print(TestLabelsidf)

## confusion matrix

cnf_matrix = confusion_matrix(TestLabelsidf, MyModelNB1.predict(TestDFidf_nolabels))
print("\nThe confusion matrix tfidf is:")
print(cnf_matrix)



skplt.metrics.plot_confusion_matrix(TestLabelsidf, MyModelNB1.predict(TestDFidf_nolabels) ,normalize=False,figsize=(12,8), cmap=plt.cm.RdGy)
plt.show()


print(metrics.classification_report(TestLabelsidf, MyModelNB1.predict(TestDFidf_nolabels)))
print(metrics.confusion_matrix(TestLabelsidf, MyModelNB1.predict(TestDFidf_nolabels)))

###SVM

from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

SVM_Model=LinearSVC(C=.1)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF))
print("\nThe SVM confusion matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TestLabelsidf, SVM_Model.predict(TestDF) ,normalize=False,figsize=(12,8), cmap=plt.cm.RdGy)
plt.show()


print(metrics.classification_report(TestLabelsidf, SVM_Model.predict(TestDF)))
print(metrics.confusion_matrix(TestLabelsidf, SVM_Model.predict(TestDF)))


###############################################################################################
###############################################################################################
###############################################################################################


## Part 2: code with the bid data 

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

# CV - regular
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
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        binary=False,
                        token_pattern=r'\b[^\d\W]+\b'
                        
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
# CV-regular
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
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        binary=False,
                        token_pattern=r'\b[^\d\W]+\b'
                        
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

### HAVE 2 DF FROM HERE READY

#########################################################################
# add Bernoli 
## This will be used to vectorize the text data for Bern
### Notice that this one is binary - which is used for Bernoulli
MyVectCV1B=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        binary=True,
                        token_pattern=r'\b[^\d\W]+\b'
                        
                        )


#########################################################################

# CV-BINARY=TRUE
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


# Now I can vectorize using my list of complete paths to my files
X_CV1B=MyVectCV1B.fit_transform(AllReviewsList)

print(MyVectCV1B.vocabulary_)

ColNames=MyVectCV1B.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CVB=pd.DataFrame(X_CV1B.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CVB = DataFrame_CVB.rename(MyDict, axis = "index")
DataFrame_CVB.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CVB.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_CVB.columns:
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
        DataFrame_CVB=DataFrame_CVB.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CVB=DataFrame_CVB.drop([nextcol], axis=1)
        

DataFrame_CVB1 = DataFrame_CVB.reset_index()
#print(DataFrame_CV)
lie_df_b = DataFrame_CVB1.copy()
print(lie_df_b) # for models 


# SECOND DF - SENTIMENT AND REVIEWS
# CV-BINARY
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

MyVectCV2B=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        binary=True,
                        token_pattern=r'\b[^\d\W]+\b'
                        
                        )
    


# Now I can vectorize using my list of complete paths to my files
X_CV2B=MyVectCV2B.fit_transform(AllReviewsList)

print(MyVectCV2B.vocabulary_)

ColNames=MyVectCV2B.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV2B=pd.DataFrame(X_CV2B.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CV2B = DataFrame_CV2B.rename(MyDict, axis = "index")
DataFrame_CV2B.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CV2B.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_CV2B.columns:
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
        DataFrame_CV2B=DataFrame_CV2B.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV2B=DataFrame_CV2B.drop([nextcol], axis=1)
        

DataFrame_CV3B = DataFrame_CV2B.reset_index()
#print(DataFrame_CV)
sentiment_df_b = DataFrame_CV3B.copy()
print(sentiment_df_b) # for models 





#########################################################################
# add TFIDF
## This will be used to vectorize the text data for Bern
### Notice that this one is binary - which is used for Bernoulli
MyVectTF=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        token_pattern=r'\b[^\d\W]+\b'
                        
                        )


#######################################################################

# TFIDF
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


# Now I can vectorize using my list of complete paths to my files
X_TF=MyVectTF.fit_transform(AllReviewsList)

print(MyVectTF.vocabulary_)

ColNames=MyVectTF.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_TF=pd.DataFrame(X_TF.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_TF = DataFrame_TF.rename(MyDict, axis = "index")
DataFrame_TF.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_TF.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_TF.columns:
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
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
        

DataFrame_TF1 = DataFrame_TF.reset_index()
#print(DataFrame_CV)
lie_df_tf = DataFrame_TF1.copy()
print(lie_df_tf) # for models 


# SECOND DF - SENTIMENT AND REVIEWS
# TFIDF
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

MyVectTF1=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=100,
                        lowercase= True,
                        ngram_range=(1, 1), 
                        analyzer='word', 
                        max_df=1.0, # ignore terms w document freq strictly > threshold 
                        min_df=1, 
                        token_pattern=r'\b[^\d\W]+\b'
                        
                        )

    


# Now I can vectorize using my list of complete paths to my files
X_TF1=MyVectTF1.fit_transform(AllReviewsList)

print(MyVectTF1.vocabulary_)

ColNames=MyVectTF1.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_TF2=pd.DataFrame(X_TF1.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_TF2 = DataFrame_TF2.rename(MyDict, axis = "index")
DataFrame_TF2.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_TF2.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in DataFrame_TF2.columns:
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
        DataFrame_CV2B=DataFrame_CV2B.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_TF2=DataFrame_TF2.drop([nextcol], axis=1)
        

DataFrame_TF3 = DataFrame_TF2.reset_index()
#print(DataFrame_CV)
sentiment_df_tf = DataFrame_TF3.copy()
print(sentiment_df_tf) # for models 

##########################################################################################

## Now we have 6 data ftrames
# cv
print("Count Vectorizer DFs: ")
print(lie_df)
print(sentiment_df)

# cv-binary
print("Count Vectorizer Bernoulli DFs: ")
print(lie_df_b)
print(sentiment_df_b)

# tfidf
print("TFIDF Vectorizer DFs: ")
print(lie_df_tf)
print(sentiment_df_tf)

##########################################################################################
### Train/test split 
# CV 
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

skplt.metrics.plot_confusion_matrix(TestLabelsLie, Prediction,normalize=False,figsize=(12,8), cmap=plt.cm.pink)
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

skplt.metrics.plot_confusion_matrix(TestLabelsSent, Prediction,normalize=False,figsize=(12,8), cmap=plt.cm.pink)
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

##################################################################################################
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

skplt.metrics.plot_confusion_matrix(TestLabelsLie, PredictionB,normalize=False,figsize=(12,8), cmap=plt.cm.Greys)
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

skplt.metrics.plot_confusion_matrix(TestLabelsSent, PredictionB1 ,normalize=False,figsize=(12,8), cmap=plt.cm.Greys)
plt.show()
plt.tight_layout()

print(np.round(BernModel1.predict_proba(TestSent),2))

print(metrics.classification_report(TestLabelsSent, PredictionB1))
print(metrics.confusion_matrix(TestLabelsSent, PredictionB1))

# top feat
print("Try this:")        
rev_important_features(MyVectCV2B,BernModel, 20)
rev_important_features(MyVectCV2B,BernModel1, 20)

#####################################################################################
### Train/test split 
# TFIDF
# Train test split -  LIE DF
## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split

TrainLieTF, TestLieTF = train_test_split(lie_df_tf, test_size=0.3)

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainLieTF)
print("\nThe testing set is:")
print(TestLieTF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabelsLieTF=TestLieTF["LABEL"]
#print(TestLabels)
## remove labels
## Make a copy of TestLie
CopyTestDFLieTF=TestLieTF.copy()
TestLieTF = TestLieTF.drop(["LABEL"], axis=1)
print(TestLieTF)

## DF seperate TRAIN SET from the labels
TrainLieTF_nolabels=TrainLieTF.drop(["LABEL"], axis=1)
#print(TrainDF_nolabels)
TrainLabelsLieTF=TrainLieTF["LABEL"]
#print(TrainLabels)

# Train test split -  SENTIMENT DF

TrainSentTF, TestSentTF = train_test_split(sentiment_df_tf, test_size=0.3)

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainSentTF)
print("\nThe testing set is:")
print(TestSentTF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabelsSentTF=TestSentTF["LABEL"]
#print(TestLabels)
## remove labels
## Make a copy of TestLie
CopyTestDFSentTF=TestSentTF.copy()
TestSentTF = TestSentTF.drop(["LABEL"], axis=1)
print(TestSentTF)

## DF seperate TRAIN SET from the labels
TrainSentTF_nolabels=TrainSentTF.drop(["LABEL"], axis=1)
#print(TrainSent_nolabels)
TrainLabelsSentTF=TrainSentTF["LABEL"]
#print(TrainLabels)


### Naive Bayes

## LIE DF

from sklearn.naive_bayes import MultinomialNB

MyModelNB01= MultinomialNB()


MyModelNB01.fit(TrainLieTF_nolabels, TrainLabelsLieTF)
Prediction = MyModelNB01.predict(TestLieTF)

print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabelsLieTF)

## confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(TestLabelsLieTF, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
#plt.imshow(cnf_matrix, cmap ='binary')

skplt.metrics.plot_confusion_matrix(TestLabelsLieTF, Prediction,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
plt.show()

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB01.predict_proba(TestLieTF),2))

from sklearn import metrics

print(metrics.classification_report(TestLabelsLieTF, Prediction))
print(metrics.confusion_matrix(TestLabelsLieTF, Prediction))

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
rev_important_features(MyVectTF,MyModelNB01, 20)


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

MyModelNB3= MultinomialNB()


MyModelNB3.fit(TrainSentTF_nolabels, TrainLabelsSentTF)
Prediction = MyModelNB3.predict(TestSentTF)

print("\nThe prediction from NB TFis:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabelsSentTF)

## confusion matrix
cnf_matrix = confusion_matrix(TestLabelsSentTF, Prediction)
print("\nThe confusion matrix TF is:")
print(cnf_matrix)
#plt.imshow(cnf_matrix, cmap ='binary')

skplt.metrics.plot_confusion_matrix(TestLabelsSentTF, Prediction,normalize=False,figsize=(12,8), cmap=plt.cm.gray_r)
plt.show()

### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB3.predict_proba(TestSentTF),2))

print(metrics.classification_report(TestLabelsSentTF, Prediction))
print(metrics.confusion_matrix(TestLabelsSentTF, Prediction))

print("FOR POS:")
FeatureRanks=sorted(zip(MyModelNB3.coef_[0],MyVectTF.get_feature_names()))
Ranks1=FeatureRanks[-10:]
## Make them unique
Ranks1 = list(set(Ranks1))
print(Ranks1)

print("\n\nFOR NEG:")

FeatureRanks=sorted(zip(MyModelNB3.coef_[0],MyVectTF.get_feature_names()),reverse=True)
Ranks2=FeatureRanks[-10:]
## Make them unique
Ranks2 = list(set(Ranks2))
print(Ranks2)

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

skplt.metrics.plot_confusion_matrix(TestLabelsLie, SVM_Model.predict(TestLie) ,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
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

skplt.metrics.plot_confusion_matrix(TestLabelsSent, SVM_Model1.predict(TestSent) ,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
plt.show()

print(metrics.classification_report(TestLabelsSent, SVM_Model1.predict(TestSent)))
print(metrics.confusion_matrix(TestLabelsSent, SVM_Model1.predict(TestSent)))


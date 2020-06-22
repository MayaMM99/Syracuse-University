# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:10:16 2020

@author: Maya 

HW 7 - Compare MNB and SVM 
IMDB Movie Review Dataset
"""


# Textmining Naive Bayes Example

import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import sklearn
import warnings
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import codecs
from sklearn import feature_extraction

#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer but with tf-idf norm

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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn import metrics


## Visualization
import matplotlib.pyplot as plt
#plt.style.use(u'seaborn-white')
import seaborn as sns
#sns.set(style="darkgrid")
sns.set(font_scale=1.3)

#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 


STEMMER=PorterStemmer()
print(STEMMER.stem("singer"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words


import string
import numpy as np

## Load the data 
df = pd.read_csv("C:\\Users\\aivii\\programsmm\\HW7_736\\IMDBDataset.csv")
print(df.head())

# Organize the column names 
df=df[['sentiment', 'review']]

#print(df.head())

df = df.rename(columns = {"sentiment":"LABEL"})
print(df.head())

print(df.describe())

print(df['LABEL'].value_counts())

## Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

## Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

## Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
## Apply function on review column
df['review']=df['review'].apply(denoise_text)

## Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

## Apply function on review column
df['review']=df['review'].apply(remove_special_characters)



####################################################################
# Explore the data - COMMENT SO IT RUN FASTER
sns.factorplot(x="LABEL", data=df, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")
plt.style.use(u'seaborn-white')
plt.show();

# See how they look 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['review'])

## Check most common words
from collections import Counter
Counter(" ".join(df["review"]).lower().split()).most_common(20)

## Check number of word in the sentence
plt.figure()
plt.hist(df['review'].str.split().apply(len).value_counts())
plt.xlabel('number of words in sentence')
plt.ylabel('frequency')
plt.title('Words occurrence frequency')

print('The maximum length of a sentence is: ',np.max(df['review'].str.split().apply(len).value_counts()))
print('The average lenth of a sentence is: ', np.average(df['review'].str.split().apply(len).value_counts()))

##########################################################################
## Get just 50% of the data 
## Generating one row  
rows = df.sample(frac =.50) 
  
## Checking if sample is 0.50 times data or not 
  
if (0.50*(len(df))== len(rows)): 
    print( "Cool") 
    print(len(df), len(rows)) 
  
## Display 
print(rows)
print("The shape of the new df with only 50% data is: ")
print(rows.shape)

# Check the distribution of the labels in the new df
sns.factorplot(x="LABEL", data=rows, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
plt.style.use(u'seaborn-white')
plt.show();

# Save the new df with 50% data 
rows.to_csv(r'C:\Users/aivii\programsmm\HW7_736\rows50.csv', index = False)

#print(df.head())
##########################################################################

#3 Grab the new data 
RawfileName0="C:/Users/aivii/programsmm/HW7_736/rows50.csv"

## This file has a header. 
## It has "LABEL" and "review" on the first row.

## We will create a list of labels and a list of reviews
AllReviewsList=[]
AllLabelsList=[]

with open(RawfileName0,'r', encoding="utf8") as FILE:
    FILE.readline() # skip header line - skip row 1
    ## This reads the line and so does nothing with it
    for row in FILE:
        NextLabel,NextReview=row.split(",", 1)
        #print(Label)
        #print(Review)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel)

#print(AllReviewsList)
#print(AllLabelsList) # all the labels 

## Additional clean up 
REPLACE_NO_SPACE = re.compile("()|(%)|(\.+)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)|(\*+)|(\.)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

AllReviewList = preprocess_reviews(AllReviewsList)
## Check the first 5 
#print(AllReviewList[5]) # looks clean 

## Have to deal with the spot words

#######################################################################
## Define the vectorizers
# no point removing them now
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
mine_stop_words = ['in', 'of', 'at', 'a', 'the','movie','film','actor', 'actors', 'also', 'in', 'im']
extend_stop_words = english_stop_words+mine_stop_words 


MyVect_CV=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        max_features=100,
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )


MyVect_STEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        max_features=100,
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )


MyVect_IFIDF=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        lowercase = True,
                        max_features=100,
                        #binary=True
                        )

MyVect_IFIDF_STEM=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        tokenizer=MY_STEMMER,
                        max_features=100,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )

###################################
## Add two for Bernouli 
MyVect_CVB=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        max_features=100,
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True, 
                        binary = True
                        )

MyVect_CVB_STEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        max_features=100,
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True, 
                        binary = True
                        )
                        


#########################################################################
## Create 4 new empty data frames
"""FinalDF_CV = pd.DataFrame()
FinalDF_STEM=pd.DataFrame()
FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF_STEM=pd.DataFrame()"""

####
##+2 for Bernouli
#FinalDF_B = pd.DataFrame()
#FinalDF_B_STEM = pd.DataFrame()

## MyVect_STEM  and MyVect_IFIDF and MyVect_IFIDF_STEM and just countvect 
X0=MyVect_CV.fit_transform(AllReviewsList)
X1=MyVect_STEM.fit_transform(AllReviewsList)
X2=MyVect_IFIDF.fit_transform(AllReviewsList)
X3=MyVect_IFIDF_STEM.fit_transform(AllReviewsList)
X4=MyVect_CVB.fit_transform(AllReviewsList)
X5=MyVect_CVB_STEM.fit_transform(AllReviewsList)



ColumnNames0=MyVect_CV.get_feature_names()
ColumnNames1=MyVect_STEM.get_feature_names()
ColumnNames2=MyVect_IFIDF.get_feature_names()
ColumnNames3=MyVect_IFIDF_STEM.get_feature_names()
ColumnNames4=MyVect_CVB.get_feature_names()
ColumnNames5=MyVect_CVB_STEM.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
FinalDF_CV =pd.DataFrame(X0.toarray(),columns=ColumnNames0)
FinalDF_STEM=pd.DataFrame(X1.toarray(),columns=ColumnNames1)
FinalDF_TFIDF=pd.DataFrame(X2.toarray(),columns=ColumnNames2)
FinalDF_TFIDF_STEM=pd.DataFrame(X3.toarray(),columns=ColumnNames3)
#builderB=pd.DataFrame(X2.toarray(),columns=ColumnNames4)
#builderBS=pd.DataFrame(X3.toarray(),columns=ColumnNames5)

########################################################################################
#1) CV- no stemmed

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

#print("MY DICT", MyDict)

FinalDF_CV = FinalDF_CV.rename(MyDict, axis = "index")
FinalDF_CV.index.name = 'LABEL'

## Drop/remove columns not wanted
#print(FinalDF_CV.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in FinalDF_CV.columns:
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
        FinalDF_CV=FinalDF_CV.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        FinalDF_CV=FinalDF_CV.drop([nextcol], axis=1)
        

FinalDF_CV1 = FinalDF_CV.reset_index()
#print(FinalDF_CV)
cv_df = FinalDF_CV1.copy()
print("CpuntVectorizer df: ")
print(cv_df) # for models 

################################
#2) CV stemmed

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

#print("MY DICT", MyDict)

FinalDF_STEM = FinalDF_STEM.rename(MyDict, axis = "index")
FinalDF_STEM.index.name = 'LABEL'

## Drop/remove columns not wanted
#print(FinalDF_STEM.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in FinalDF_STEM.columns:
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
        FinalDF_CV=FinalDF_CV.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        FinalDF_STEM=FinalDF_STEM.drop([nextcol], axis=1)
        

FinalDF_STEM1 = FinalDF_STEM.reset_index()
#print(FinalDF_CV)
cvSTEM_df = FinalDF_STEM1.copy()
print("CountVect stemmed df:")
print(cvSTEM_df) # for models 

################################

#3) TFIDF NOT STEMMED

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

#print("MY DICT", MyDict)

FinalDF_TFIDF = FinalDF_TFIDF.rename(MyDict, axis = "index")
FinalDF_TFIDF.index.name = 'LABEL'

## Drop/remove columns not wanted
#print(FinalDF_STEM.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in FinalDF_TFIDF.columns:
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
        FinalDF_TFIDF=FinalDF_TFIDF.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        FinalDF_TFIDF=FinalDF_TFIDF.drop([nextcol], axis=1)
        

FinalDF_TFIDF1 = FinalDF_TFIDF.reset_index()
#print(FinalDF_TFIDF
tfidf_df = FinalDF_TFIDF1.copy()
print("TFIDF NOT stemmed df:")
print(tfidf_df) # for models 

##########################################
#4) TFIDF STEMMED

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

#print("MY DICT", MyDict)

FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.rename(MyDict, axis = "index")
FinalDF_TFIDF_STEM.index.name = 'LABEL'

## Drop/remove columns not wanted
#print(FinalDF_STEM.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in FinalDF_TFIDF_STEM.columns:
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
        FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.drop([nextcol], axis=1)
        

FinalDF_TFIDF_STEM1 =FinalDF_TFIDF_STEM.reset_index()
#print(FinalDF_TFIDF_STEM)
tfidfSTEM_df = FinalDF_TFIDF_STEM1.copy()
print("TFIDF stemmed df:")
print(tfidfSTEM_df) # for models 

## View all
print(cv_df)
print(cvSTEM_df)
print(tfidf_df)
print(tfidfSTEM_df)

## Create the testing set - grab a sample from the training set. 
## Be careful. Notice that right now, our train set is sorted by label.
## If your train set is large enough, you can take a random sample.
from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(cv_df, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(cvSTEM_df, test_size=0.3)
TrainDF3, TestDF3 = train_test_split(tfidf_df, test_size=0.3)
TrainDF4, TestDF4 = train_test_split(tfidfSTEM_df, test_size=0.3)


### OK - at this point we have Train and Test data for the text data
## Of course, this can be updated to work from sentiment (like POS and NEG)
## and can be update for multiple folders or one folder..

###############################################
## For all three DFs - separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
### TEST ---------------------
Test1Labels=TestDF1["LABEL"]
Test2Labels=TestDF2["LABEL"]
Test3Labels=TestDF3["LABEL"]
Test4Labels=TestDF4["LABEL"]
print(Test2Labels)

## remove labels
TestDF1 = TestDF1.drop(["LABEL"], axis=1)
TestDF2 = TestDF2.drop(["LABEL"], axis=1)
TestDF3 = TestDF3.drop(["LABEL"], axis=1)
TestDF4 = TestDF4.drop(["LABEL"], axis=1)
print(TestDF1)

## TRAIN ----------------------------
Train1Labels=TrainDF1["LABEL"]
Train2Labels=TrainDF2["LABEL"]
Train3Labels=TrainDF3["LABEL"]
Train4Labels=TrainDF4["LABEL"]

print(Train2Labels)

## remove labels
TrainDF1 = TrainDF1.drop(["LABEL"], axis=1)
TrainDF2 = TrainDF2.drop(["LABEL"], axis=1)
TrainDF3 = TrainDF3.drop(["LABEL"], axis=1)
TrainDF4 = TrainDF4.drop(["LABEL"], axis=1)
print(TrainDF3)

###################################################
### Naive Bayes -  coment so it run faster

from sklearn.naive_bayes import MultinomialNB

MyModelNB1= MultinomialNB()
MyModelNB2= MultinomialNB()
MyModelNB3= MultinomialNB()
MyModelNB4= MultinomialNB()
## When you look up this model, you learn that it wants the 

## Run on all three Dfs.................
MyModelNB1.fit(TrainDF1, Train1Labels)
MyModelNB2.fit(TrainDF2, Train2Labels)
MyModelNB3.fit(TrainDF3, Train3Labels)
MyModelNB4.fit(TrainDF4, Train4Labels)

Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
Prediction3 = MyModelNB3.predict(TestDF3)
Prediction4 = MyModelNB4.predict(TestDF4)


print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)

print("\nThe prediction from NB is:")
print(Prediction3)
print("\nThe actual labels are:")
print(Test3Labels)

print("\nThe prediction from NB is:")
print(Prediction4)
print("\nThe actual labels are:")
print(Test4Labels)

## confusion matrix
from sklearn.metrics import confusion_matrix

cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

skplt.metrics.plot_confusion_matrix(Test1Labels, Prediction1,normalize=False,figsize=(12,8), cmap=plt.cm.PuBu)
plt.show()


cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

skplt.metrics.plot_confusion_matrix(Test2Labels, Prediction2,normalize=False,figsize=(12,8), cmap=plt.cm.PuBu)
plt.show()

cnf_matrix3 = confusion_matrix(Test3Labels, Prediction3)
print("\nThe confusion matrix is:")
print(cnf_matrix3)

skplt.metrics.plot_confusion_matrix(Test3Labels, Prediction3,normalize=False,figsize=(12,8), cmap=plt.cm.PuBu)
plt.show()

cnf_matrix4 = confusion_matrix(Test4Labels, Prediction4)
print("\nThe confusion matrix is:")
print(cnf_matrix4)

skplt.metrics.plot_confusion_matrix(Test4Labels, Prediction4,normalize=False,figsize=(12,8), cmap=plt.cm.PuBu)
plt.show()


### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB1.predict_proba(TestDF1),2))

print(metrics.classification_report(Test1Labels, Prediction1))
print(metrics.confusion_matrix(Test1Labels, Prediction1))
####
print(np.round(MyModelNB2.predict_proba(TestDF2),2))

print(metrics.classification_report(Test2Labels, Prediction2))
print(metrics.confusion_matrix(Test2Labels, Prediction2))
#####
print(np.round(MyModelNB3.predict_proba(TestDF3),2))

print(metrics.classification_report(Test3Labels, Prediction3))
print(metrics.confusion_matrix(Test3Labels, Prediction3))
#####
print(np.round(MyModelNB4.predict_proba(TestDF4),2))

print(metrics.classification_report(Test4Labels, Prediction4))
print(metrics.confusion_matrix(Test4Labels, Prediction4))

###################################################################################
import matplotlib.pyplot as plt
## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
## Define a function to visualize the TOP words (variables)
def plot_coefficients(MODEL=MyModelNB1, COLNAMES=TrainDF1.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["#6d6875"if c < -2 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()

def plot_coefficients(MODEL=MyModelNB2, COLNAMES=TrainDF2.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["#6d6875" if c < -2 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()


def plot_coefficients(MODEL=MyModelNB3, COLNAMES=TrainDF3.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["#6d6875"if c < -2 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()

def plot_coefficients(MODEL=MyModelNB4, COLNAMES=TrainDF4.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["#6d6875"if c < -2 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()


    
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
rev_important_features(MyVect_CV,MyModelNB1, 10)
rev_important_features(MyVect_STEM,MyModelNB2, 10)
rev_important_features(MyVect_IFIDF,MyModelNB3, 10)
rev_important_features(MyVect_IFIDF_STEM,MyModelNB4, 10)

#######################################################################################
## SVM 
from sklearn.svm import LinearSVC
## linear 

# change the df here
TRAIN= TrainDF1
TRAIN_Labels= Train1Labels
TEST= TestDF1
TEST_Labels= Test1Labels

## change the kernels here 
SVM_Model1=LinearSVC(C=0.0001)
SVM_Model1.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model1.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST) ,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
plt.show()

print(metrics.classification_report(TEST_Labels, SVM_Model1.predict(TEST)))
print(metrics.confusion_matrix(TEST_Labels, SVM_Model1.predict(TEST)))
#--------------other kernels


## RBF
SVM_Model2=sklearn.svm.SVC(C=0.0001, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model2.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model2.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST) ,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
plt.show()

print(metrics.classification_report(TEST_Labels, SVM_Model2.predict(TEST)))
print(metrics.confusion_matrix(TEST_Labels, SVM_Model2.predict(TEST)))



## POLY
SVM_Model3=sklearn.svm.SVC(C=0.0001, kernel='poly',degree=2,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(TRAIN, TRAIN_Labels)

print("SVM prediction:\n", SVM_Model3.predict(TEST))
print("Actual:")
print(TEST_Labels)

SVM_matrix = confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

skplt.metrics.plot_confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST) ,normalize=False,figsize=(12,8), cmap=plt.cm.gray)
plt.show()

print(metrics.classification_report(TEST_Labels, SVM_Model3.predict(TEST)))
print(metrics.confusion_matrix(TEST_Labels, SVM_Model3.predict(TEST)))

################################################################################

"""def get_features(vec, thingy):
    feature_ranks = sorted(zip(thingy.coef_[0], vec.get_feature_names()))

    very_negative_10 = feature_ranks[-10:]
#     print("Very negative words")
    vn = []
    for i in range(0, len(very_negative_10)):
        vn.append(very_negative_10[i])
#         print(very_negative_10[i])
    df_neg = pd.DataFrame(vn)
    show_feature_table(df_neg, 'Most Negative Words')
#     print()

    not_very_negative_10 = feature_ranks[:10]
#     print("Not very negative words")
    nvn = []
    for i in range(0, len(not_very_negative_10)):
#         print(not_very_negative_10[i])
        nvn.append(not_very_negative_10[i])
    df_pos = pd.DataFrame(nvn)
#     print(df_n)
    
    show_feature_table(df_pos, 'Most Positive Words')
    
get_features(MyVect_CV, SVM_Model1)"""  

"""print("FOR HIKES:")
FeatureRanks=sorted(zip(MyModelNB.coef_[0],MyVect5.get_feature_names()))
Ranks1=FeatureRanks[-20:]
## Make them unique
Ranks1 = list(set(Ranks1))
print(Ranks1)

print("\n\nFOR DOGS:")

FeatureRanks=sorted(zip(MyModelNB.coef_[0],MyVect5.get_feature_names()),reverse=True)
Ranks2=FeatureRanks[-20:]
## Make them unique
Ranks2 = list(set(Ranks2))
print(Ranks2)"""

# linear c= 0.0001
def plot_coefficients(MODEL=SVM_Model1, COLNAMES=TrainDF1.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["#6d6875" if c < 0 else "#05668d" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()



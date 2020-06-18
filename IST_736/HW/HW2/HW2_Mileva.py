# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:50:40 2020

@author: Maya

w2 homework 2
ist 736 

"""


import nltk 
from nltk import FreqDist
import re
from textblob import TextBlob
from IPython.display import display, HTML
import warnings
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import * 
import sklearn
from sklearn.cluster import KMeans 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
##For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
sid = SentimentIntensityAnalyzer()

# Countvectorizer 
text = ["I love my dog. My dog is my best friend, he is awesome. Me and the dog go for a walk every day and my dog meet all his friends. I love  my dog`s blue eyes and that he is so good with people. I love him so much."]

# create the transform
vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
print("My vocabulary is..")
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["I love my dog. My dog is my best friend, he is awesome. Me and the dog go for a walk every day and my dog meet all his friends. I love  my dog`s blue eyes and that he is so good with people. I love him so much.",
		"I love walking outside. It is really nice and refreshing and I burn calories. I walk at least 3 miles every day with my dog. We enjoy the nice weather in Florida during out winter walks."]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())


## 1. My small examples 

MyVectorizer1=CountVectorizer(
        input='content', ## can be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
       ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer
        encoding='latin-1',
        decode_error='ignore', #{‘strict’, ‘ignore’, ‘replace’}
        strip_accents=None, # {‘ascii’, ‘unicode’, None}
        lowercase=True, 
        #preprocessor=None, 
        #tokenizer=None, 
        #stop_words='english', #string {‘english’}, list, or None (default)
        stop_words=None,
        token_pattern='(?u)\b\w\w+\b', #Regular expression denoting what constitutes a “token”
        ngram_range=(1, 1), 
        analyzer='word', 
        max_df=1.0, # ignore terms w document freq strictly > threshold 
        min_df=1, 
        max_features=None, 
        vocabulary=None, 
        binary=False, #If True, all non zero counts are set to 1
        #dtype=<class 'numpy.int64'> 
        )

# Get the list of the docs 
path = "C:\\Users\\aivii\\programsmm\\SmallTextDocs" # defining a path that i`m reusing
print("calling os...")
# Print the files in this location to make sure the location is right 
print(os.listdir(path)) # ['Dog.txt', 'Walk.txt']

# Save the results to a list 
FileNameList = os.listdir(path)
# check type
print(type(FileNameList))
print(FileNameList) # now I have a list of filenames 

# Now I need complete paths in order to use CountVectorizer
# create an empty list to start 
ListOfCompleteFilePaths=[]
ListOfJustFileNames=[]

for name in os.listdir(path):
    #C:\\Users\\aivii\\programsmm\\SmallTextDocs
    print(path+ "\\" + name)
    next=path+ "\\" + name
    
    nextnameL=name.split(".")   ##If name is Dog.txt is splits it into Dog   and txt
    nextname=nextnameL[0]   ## This gives me: Dog
    
    #print("DONE...")
    print("full list...")
    ListOfCompleteFilePaths.append(next)
    ListOfJustFileNames.append(nextname)

## CountVectorizers be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer

MyVect3 = CountVectorizer(input = 'filename')
#Now I can vectorize using my list of complete paths to my files 
D_DW = MyVect3.fit_transform(ListOfCompleteFilePaths)

# Print the results 
print(D_DW)

# get the features names 
ColumnNames3=MyVect3.get_feature_names()
#print(ColumnNames3)

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_DogWalk = pd.DataFrame(D_DW.toarray(),columns=ColumnNames3)
print(CorpusDF_DogWalk)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

CorpusDF_DogWalk = CorpusDF_DogWalk.rename(MyDict, axis="index")
print(CorpusDF_DogWalk)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(CorpusDF_DogWalk)) # we have a data frame 

# Have to convert to matrix 

# Convert DataFrame to matrix
MyMatrixDogWalk = CorpusDF_DogWalk.values
## Check it

print(type(MyMatrixDogWalk)) # array 
print(MyMatrixDogWalk)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np

kmeans_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object.fit(MyMatrixDogWalk)

# Get cluster assignment labels
labels = kmeans_object.labels_
#print(labels)
# Format results as a DataFrame
Myresults = pd.DataFrame([CorpusDF_DogWalk.index,labels]).T
print(Myresults)

### Hmmm -  these are not great results
## This is because my dataset if not clean
## I still have stopwords
## I still have useless or small words < size 3

## Let's clean it up....
## Let's start with this: 

print(CorpusDF_DogWalk)

## Let's remove our own stopwords that WE create

## Let's also remove all words of size 2 or smaller
## Finally, without using a stem package - 
## Let's combine columns with dog, dogs
## and with walk, walkss, walking

## We need to "build" this in steps.
## First, I know that I need to be able to access
## the columns...

for name in ColumnNames3:
    print(name)

# Now we can access by column name 
for name in ColumnNames3:
    print(CorpusDF_DogWalk[name])
    
name1 = "walks"
name2 = "walk"
if(name1==name2):
    print("TRUE")
else:
    print("FALSE")

name1=name1.rstrip("s")
print(name1)
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")

print("The initial column names:\n", ColumnNames3)
print(type(ColumnNames3))  ## This is a list

#### Stop word list
MyStops=["also", "and", "are", "you", "of", "let", "not", "the", "for", "why", "there", "one", "which"]   

# MAKE COPIES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CleanDF = CorpusDF_DogWalk
print("START\n",CleanDF)
## Build a new columns list
ColNames=[]

for name in ColumnNames3:
    #print("FFFFFFFF",name)
    if ((name in MyStops) or (len(name)<3)): 
        print("Dropping: ", name)
        CleanDF=CleanDF.drop([name], axis=1)
        print(CleanDF)
    else:
        ## I MUST add these new ColNames
        ColNames.append(name)

print("END\n",CleanDF)             
print("The ending column names:\n", ColNames)

for name1 in ColNames:
    for name2 in ColNames:
        if(name1 == name2):
            print("skip")
        elif(name1.rstrip("e") in name2):  ## this is good for plurals
            ## like dog and dogs, but not for hike and hiking
            ## so I will strip an "e" if there is one...
            print("combining: ", name1, name2)
            print(CorpusDF_DogWalk[name1])
            print(CorpusDF_DogWalk[name2])
            print(CorpusDF_DogWalk[name1] + CorpusDF_DogWalk[name2])
            
            ## Think about how to test this!
            ## at first, you can do this:
            ## NEW=name1+name2
            ## CleanDF[NEW]=CleanDF[name1] + CleanDF[name2]
            ## Then, before dropping any columns - print
            ## the columns and their sum to check it. 
            
            CleanDF[name1] = CleanDF[name1] + CleanDF[name2]
            
            ### Later and once everything is tested - you
            ## will include this next line of code. 
            ## While I tested everyting, I had this commented out
            ###   "******
            CleanDF=CleanDF.drop([name2], axis=1)
        
print(CleanDF.columns.values)

## Confirm that your column summing is working!

print(CleanDF["dog"])
#print(CleanDF["dogs"])
#print(CleanDF["dogdogs"])  ## this should be the sum

## AFTER
print(CleanDF)

## NOW - let's try k means again....
############################## k means ########################

# Convert DataFrame to matrix
MyMatrixClean = CleanDF.values
## Check it
print(type(MyMatrixClean))
print(MyMatrixClean)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object2 = sklearn.cluster.KMeans(n_clusters=3)
print(kmeans_object2)
#kmeans_object2.fit(MyMatrixClean)
# Get cluster assignment labels
#labels2 = kmeans_object2.labels_
#print("k-means with k = 3\n", labels2)
# Format results as a DataFrame
#Myresults2 = pd.DataFrame([CleanDF.index,labels2]).T
#print("k means RESULTS\n", Myresults2)

################# k means with k = 2 #####################

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object3 = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object3.fit(MyMatrixClean)
# Get cluster assignment labels
labels3 = kmeans_object3.labels_
print("K means with k = 2\n", labels3)
# Format results as a DataFrame
Myresults3 = pd.DataFrame([CleanDF.index,labels3]).T
print("k means RESULTS\n", Myresults3)

print("My CleanDF..")

CleanDF = CleanDF.sample(frac = 1).reset_index(drop=True)
print(CleanDF.head)

np.random.seed(140)
## sample without replacement 

train_ix = np.random.choice(CleanDF.index, 1, replace = False)
df_training = CleanDF.iloc[train_ix]
df_test = CleanDF.drop(train_ix)

print("Training set....")
print(df_training)


print("Testing set...")
print(df_test)

############################################################################################

## TWEETS 

# POS and Ngative tags
# STEP1: Read the POS files corpus into a DF1
print("Building a Vectorizer...")
myVect4 = CountVectorizer(input = 'filename',
                          analyzer = 'word',
                          stop_words = 'english',
                          token_pattern = '(?u)[a-zA-Z]+' # letters only, no numbers 
                          )
path = "C:\\Users\\aivii\\programsmm\\AIPOS"

# Create an empty list
POSListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    POSListOfCompleteFiles.append(next)

print("POS full list...")
print(POSListOfCompleteFiles) # LIST THE ALL 5

print("FIT with Vectorizer...")
P_TW = myVect4.fit_transform(POSListOfCompleteFiles)
print(type(P_TW))
print(P_TW.get_shape())

POSColumnNames = myVect4.get_feature_names()
print("Column names: ", POSColumnNames[0:10])
print("Building DF...")

POS_CorpusDF_TW = pd. DataFrame(P_TW.toarray(), columns = POSColumnNames)
print(POS_CorpusDF_TW)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(POSColumnNames)):
    MyDict[i] = POSColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

POS_CorpusDF_TW  = POS_CorpusDF_TW .rename(MyDict, axis="index")
print(POS_CorpusDF_TW)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(POS_CorpusDF_TW))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = POS_CorpusDF_TW.shape
print(lenght[0]) ## num of rows 
print(lenght[1]) ##num of columns 

## Add column
POS_CorpusDF_TW["PosORNeg"] = "P"
print(POS_CorpusDF_TW) 
# Now we have 57 columns

## Same for negative folder 
path = "C:\\Users\\aivii\\programsmm\\AINEG"


# Create an empty list
NEGListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    NEGListOfCompleteFiles.append(next)

print("NEG full list...")
print(NEGListOfCompleteFiles) # LIST THE ALL 

print("FIT with Vectorizer...")
N_TW = myVect4.fit_transform(NEGListOfCompleteFiles)
print(type(N_TW))
print(N_TW.get_shape())

NEGColumnNames = myVect4.get_feature_names()
print("Column names: ", NEGColumnNames[0:10])
print("Building DF...")

NEG_CorpusDF_TW = pd. DataFrame(N_TW.toarray(), columns = NEGColumnNames)
print(NEG_CorpusDF_TW)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(NEGColumnNames)):
    MyDict[i] = NEGColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

NEG_CorpusDF_TW  = NEG_CorpusDF_TW .rename(MyDict, axis="index")
print(NEG_CorpusDF_TW)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(NEG_CorpusDF_TW))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = NEG_CorpusDF_TW.shape
print(lenght[0]) ## num of rows
print(lenght[1]) ##num of columns

## Add column
NEG_CorpusDF_TW["PosORNeg"] = "N"
print(NEG_CorpusDF_TW)

## NEXT: Create a Complete DF with all data and shuffle - the build test and train set 
# Create new large Pos and Neg df 

result = NEG_CorpusDF_TW.append(POS_CorpusDF_TW)
print(result)
# replace Nan with 0 
# actually mean non in this case 

result = result.fillna(0)
print(result)

result = result.sample(frac = 1).reset_index(drop=True)
print(result.head)

np.random.seed(140)
## sample without replacement 

train_ix = np.random.choice(result.index, 6, replace = False)
df_training = result.iloc[train_ix]
df_test = result.drop(train_ix)

print("Training set....")
print(df_training)


print("Testing set...")
print(df_test)


############################################################################################
# Movie Reviews -  small data 

# POS and Ngative tags
# STEP1: Read the POS files corpus into a DF1
print("Building a Vectorizer...")
myVect5 = CountVectorizer(input = 'filename',
                          analyzer = 'word',
                          stop_words = 'english',
                          token_pattern = '(?u)[a-zA-Z]+',# letters only, no numbers 
                          decode_error = 'ignore'
                          )
path = "C:\\Users\\aivii\\programsmm\\POS"

# Create an empty list
POSListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    POSListOfCompleteFiles.append(next)

print("POS full list...")
print(POSListOfCompleteFiles) # LIST THE ALL 5

print("FIT with Vectorizer...")
P_MV = myVect5.fit_transform(POSListOfCompleteFiles)
print(type(P_MV))
print(P_MV.get_shape())

POSColumnNames = myVect5.get_feature_names()
print("Column names: ", POSColumnNames[0:10])
print("Building DF...")

POS_CorpusDF_MV = pd. DataFrame(P_MV.toarray(), columns = POSColumnNames)
print(POS_CorpusDF_MV)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(POSColumnNames)):
    MyDict[i] = POSColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

POS_CorpusDF_MV  = POS_CorpusDF_MV .rename(MyDict, axis="index")
print(POS_CorpusDF_MV)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(POS_CorpusDF_MV))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = POS_CorpusDF_MV.shape
print(lenght[0]) ## num of rows 
print(lenght[1]) ##num of columns 

## Add column
POS_CorpusDF_MV["PosORNeg"] = "P"
print(POS_CorpusDF_MV) 

## Same for negative folder 
path = "C:\\Users\\aivii\\programsmm\\NEG"


# Create an empty list
NEGListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    NEGListOfCompleteFiles.append(next)

print("NEG full list...")
print(NEGListOfCompleteFiles) # LIST THE ALL 

print("FIT with Vectorizer...")
N_MV = myVect5.fit_transform(NEGListOfCompleteFiles)
print(type(N_MV))
print(N_MV.get_shape())

NEGColumnNames = myVect5.get_feature_names()
print("Column names: ", NEGColumnNames[0:10])
print("Building DF...")

NEG_CorpusDF_MV = pd. DataFrame(N_MV.toarray(), columns = NEGColumnNames)
print(NEG_CorpusDF_MV)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(NEGColumnNames)):
    MyDict[i] = NEGColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

NEG_CorpusDF_MV  = NEG_CorpusDF_MV .rename(MyDict, axis="index")
print(NEG_CorpusDF_MV)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(NEG_CorpusDF_MV))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = NEG_CorpusDF_MV.shape
print(lenght[0]) ## num of rows
print(lenght[1]) ##num of columns

## Add column
NEG_CorpusDF_MV["PosORNeg"] = "N"
print(NEG_CorpusDF_MV)

## NEXT: Create a Complete DF with all data and shuffle - the build test and train set 
# Create new large Pos and Neg df 

result = NEG_CorpusDF_MV.append(POS_CorpusDF_MV)
print(result)
# replace Nan with 0 
# actually mean non in this case 

result = result.fillna(0)
print(result)

result = result.sample(frac = 1).reset_index(drop=True)
print(result.head)

np.random.seed(140)
## sample without replacement 

train_ix1 = np.random.choice(result.index, 6, replace = False)
df_trainingmv = result.iloc[train_ix1]
df_testmv = result.drop(train_ix1)

print("Training set....")
print(df_trainingmv)


print("Testing set...")
print(df_testmv)

############################################################################################
# Movie Reviews -  BIG data 
# POS and Ngative tags
# STEP1: Read the POS files corpus into a DF1
print("Building a Vectorizer...")

myVect6 = CountVectorizer(input = 'filename',
                          analyzer = 'word',
                          stop_words = 'english',
                          token_pattern = '(?u)[a-zA-Z]+', # letters only, no numbers
                          decode_error = 'ignore'
                          )

path = "C:\\Users\\aivii\\programsmm\\posBIG"

# Create an empty list
POSListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    POSListOfCompleteFiles.append(next)

print("POS full list...")
print(POSListOfCompleteFiles) # LIST THE ALL 5

print("FIT with Vectorizer...")
P_Bmv = myVect6.fit_transform(POSListOfCompleteFiles)
print(type(P_Bmv))
print(P_Bmv.get_shape())

POSColumnNames = myVect6.get_feature_names()
print("Column names: ", POSColumnNames[0:10])
print("Building DF...")

POS_CorpusDF_Bmv = pd. DataFrame(P_Bmv.toarray(), columns = POSColumnNames)
print(POS_CorpusDF_Bmv)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(POSColumnNames)):
    MyDict[i] = POSColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

POS_CorpusDF_Bmv  = POS_CorpusDF_Bmv .rename(MyDict, axis="index")
print(POS_CorpusDF_Bmv)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(POS_CorpusDF_Bmv))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = POS_CorpusDF_Bmv.shape
print(lenght[0]) ## num of rows 
print(lenght[1]) ##num of columns 

## Add column
POS_CorpusDF_Bmv["PosORNeg"] = "P"
print(POS_CorpusDF_Bmv) 
# Now we have 57 columns

## Same for negative folder 
path = "C:\\Users\\aivii\\programsmm\\negBIG"


# Create an empty list
NEGListOfCompleteFiles = []

for name in os.listdir(path):
    print(path+ "\\" + name)
    next = path + "\\" +name
    NEGListOfCompleteFiles.append(next)

print("NEG full list...")
print(NEGListOfCompleteFiles) # LIST THE ALL 

print("FIT with Vectorizer...")
N_Bmv = myVect6.fit_transform(NEGListOfCompleteFiles)
print(type(N_Bmv))
print(N_Bmv.get_shape())

NEGColumnNames = myVect6.get_feature_names()
print("Column names: ", NEGColumnNames[0:10])
print("Building DF...")

NEG_CorpusDF_Bmv = pd. DataFrame(N_Bmv.toarray(), columns = NEGColumnNames)
print(NEG_CorpusDF_Bmv)

## Now update the row names
MyDict={} # create an empty dictionary
for i in range(0, len(NEGColumnNames)):
    MyDict[i] = NEGColumnNames[i]

print("MY DICT:", MyDict) # print the items in my dictionary 

NEG_CorpusDF_Bmv  = NEG_CorpusDF_Bmv.rename(MyDict, axis="index")
print(NEG_CorpusDF_Bmv)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have

print(type(NEG_CorpusDF_Bmv))

## Now we need to add column for pos and neg
## I will call it PosORNeg and because all of these 
## are pos , I will fill it with P
#DataFrame.inset(loc, column, value, allow_duplicates = False)
lenght = NEG_CorpusDF_Bmv.shape
print(lenght[0]) ## num of rows
print(lenght[1]) ##num of columns

## Add column
NEG_CorpusDF_Bmv["PosORNeg"] = "N"
print(NEG_CorpusDF_Bmv)

## NEXT: Create a Complete DF with all data and shuffle - the build test and train set 
# Create new large Pos and Neg df 

result = NEG_CorpusDF_Bmv.append(POS_CorpusDF_Bmv)
print(result)
# replace Nan with 0 
# actually mean non in this case 

result = result.fillna(0)
print(result)

result = result.sample(frac = 1).reset_index(drop=True)
print(result.head)

np.random.seed(140)
## sample without replacement 

train_ix1 = np.random.choice(result.index, 1120, replace = False)
df_trainingBmv = result.iloc[train_ix1]
df_testBmv = result.drop(train_ix1)

print("Training set....")
print(df_trainingBmv)


print("Testing set...")
print(df_testBmv)


################################################################################################
# Second Way 
# Tokenization
text = "I love my dog. My dog is my best friend, he is awesome. Me and the dog go for a walk every day and my dog meet all his friends. I love  my dog`s blue eyes and that he is so good with people. I love him so much."
texttokens = nltk.word_tokenize(text)
print(len(texttokens))
print(texttokens)

# Freq Distribution
fdist = FreqDist(texttokens)
print(fdist)

fdist.plot(28)

# Punctuation 
# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False
alphwords = [w for w in texttokens if not alpha_filter(w)]

# Stop words 
from nltk.corpus import stopwords
set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

filtered_sentence = [w for w in alphwords if not w in stop_words]
filtered_sentence = [] 
  
for w in alphwords: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(texttokens) 
print(filtered_sentence) 

# create another plot 
fdist1 = FreqDist(filtered_sentence)
print(fdist1)

fdist1.plot(22)


#################################################################

# Define function to grab the positive and negative reviews
def get_data_from_files(path):
    directory = os.listdir(path)
    results = []
    for file in directory:
        f=open(path+file)
        results.append(f.read())
        f.close()
    return results


neg_tw = get_data_from_files('AINEG/')
pos_tw = get_data_from_files('AIPOS/')

neg_mv = get_data_from_files('NEG/')
pos_mv = get_data_from_files('POS/')

neg_BIGmv = get_data_from_files('negBIG/')
pos_BIGmv = get_data_from_files('posBIG/')

print("Tweets...")
print(len(neg_tw)) #5
print(len(pos_tw)) #5

print("Movie Reviews...")
print(len(neg_mv)) #5
print(len(pos_mv)) #5

print("BIG Movie Reviews...")
print(len(neg_BIGmv))
print(len(pos_BIGmv))

#####################################
## VADER
sid = SentimentIntensityAnalyzer()

# 1. Movie reviews - small
negative = os.listdir('NEG/')
positive = os.listdir('POS/')
positive_alltext = []
for file in positive:
    f=open('POS/'+file)
    content=f.read()
    positive_alltext.append(content)
    f.close()

negative_alltext = []
for file in negative:
    f=open('NEG/'+file)
    content=f.read()
    negative_alltext.append(content)
    f.close()
    
## pos
for sentence in positive_alltext:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end="")
        print()   
        
## neg 
for sentence in negative_alltext:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end="")
        print()

# 2. Tweets 
negative = os.listdir('AINEG/')
positive = os.listdir('AIPOS/')
positive_alltext = []
for file in positive:
    f=open('AIPOS/'+file)
    content=f.read()
    positive_alltext.append(content)
    f.close()

negative_alltext = []
for file in negative:
    f=open('AINEG/'+file)
    content=f.read()
    negative_alltext.append(content)
    f.close()

## pos
for sentence in positive_alltext:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end="")
        print()   
        
## neg 
for sentence in negative_alltext:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end="")
        print()

# 3. Movie Reviews big
        
######################################################
## VADER - with function

def get_pn(num):
    return 'neg' if num < 0 else 'pos'

def get_vader_scores(array, label):
    vader_array = []
    for sentence in array:
        ss = sid.polarity_scores(sentence)
        vader_array.append({'label': label,
                            'prediction': get_pn(ss['compound']),
                            'compound': ss['compound'], 
                            'excerpt': sentence[:50]})
    return vader_array

# 1.Tweets

df_n = pd.DataFrame(get_vader_scores(neg_tw, 'neg'))
df_p = pd.DataFrame(get_vader_scores(pos_tw, 'pos'))

df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n)
display(df_p)

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))


# 2.Movie - small

df_n = pd.DataFrame(get_vader_scores(neg_mv, 'neg'))
df_p = pd.DataFrame(get_vader_scores(pos_mv, 'pos'))

df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n)
display(df_p)

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))

# 3. Movie - big

df_n = pd.DataFrame(get_vader_scores(neg_BIGmv, 'neg'))
df_p = pd.DataFrame(get_vader_scores(pos_BIGmv, 'pos'))

df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n[:5])
display(df_p[:5])

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))

#######################################################
## TextBlob
def get_pn(num):
    return 'neg' if num < 0 else 'pos'

def get_sentiment(array, label):
    blobs = [[TextBlob(text), text] for text in array]
    return ([{'label': label,
              'prediction': get_pn(obj.sentiment.polarity),
              'sentiment': obj.sentiment.polarity,
              'length': len(text), 
              'excerpt': text[:50], 
              'tags': obj.tags} for obj,text in blobs])
    
# A sentiment lexicon can be used to discern objective facts from subjective opinions in text. 
# Each word in the lexicon has scores for: 
# 1) polarity: negative vs. positive (-1.0 => +1.0) 
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0) 
# 3) intensity: modifies next word? (x0.5 => x2.0)
 
## Tweets 

df_n = pd.DataFrame(get_sentiment(neg_tw, 'neg'))
df_p = pd.DataFrame(get_sentiment(pos_tw, 'pos'))

df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n)
display(df_p)

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))

## Movie - small
df_n = pd.DataFrame(get_sentiment(neg_mv, 'neg'))
df_p = pd.DataFrame(get_sentiment(pos_mv, 'pos'))

import numpy as np
df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n)
display(df_p)

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))

## Movie - big
df_n = pd.DataFrame(get_sentiment(neg_BIGmv, 'negBIG'))
df_p = pd.DataFrame(get_sentiment(pos_BIGmv, 'posBIG'))

import numpy as np
df_n['accurate'] = np.where(df_n['label'] == df_n['prediction'], 'yes', 'no')
df_p['accurate'] = np.where(df_p['label'] == df_p['prediction'], 'yes', 'no')

display(df_n)
display(df_p)

sum_correct_n = (df_n['accurate']=='yes').sum()
sum_correct_p = (df_p['accurate']=='yes').sum()

print('CORRECT PREDICT FALSE:', sum_correct_n, 'out of', len(df_n), sum_correct_n/len(df_n))
print('CORRECT PREDICT TRUE:', sum_correct_p, 'out of', len(df_p), sum_correct_p/len(df_p))
# twwets
#display(pd.DataFrame(get_sentiment(neg_tw, 'neg')))
#display(pd.DataFrame(get_sentiment(pos_tw, 'pos')))

# movies
#display(pd.DataFrame(get_sentiment(neg_mv, 'neg')))
#display(pd.DataFrame(get_sentiment(pos_mv, 'pos')))

##########################################################
## NLTK Wwith NB 

def get_tokens(sentence): # get all the tokens
    tokens = word_tokenize(sentence)
    clean_tokens = [word.lower() for word in tokens if word.isalpha()]
    return clean_tokens

def get_nltk_train_test(array, label, num_train):
    tokens = [get_tokens(sentence) for sentence in array]
    docs = [(sent, label) for sent in tokens]
    train_docs = docs[:num_train]
    test_docs = docs[num_train:len(array)]
    return [train_docs, test_docs]


def get_nltk_NB(NEG_DATA, POS_DATA, num_train):
    train_neg, test_neg = get_nltk_train_test(NEG_DATA, 'neg', num_train)
    train_pos, test_pos = get_nltk_train_test(POS_DATA, 'pos', num_train)

    training_docs = train_neg + train_pos
    testing_docs = test_neg + test_pos

    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
    
    #results = []
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key,value))

## Tweets
get_nltk_NB(neg_tw, pos_tw, 3)

## Movie - small
get_nltk_NB(neg_mv, pos_mv, 3)

## Movie - big
get_nltk_NB(neg_BIGmv, pos_BIGmv, 450)
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:34:37 2020

@author: Maya
HW3 - Dirty Data 
"""


################################
### This program read dirty data from csv file and trasform it into labeled dataframe
################################

## Textmining Naive Bayes Example
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

## Document 1

RawfileName0="C:/Users/aivii/programsmm/HW3_736/RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE.csv"

## This file has a header. 
## It has "setinment" and "review" on the first row.

AllReviewsList=[]
AllLabelsList=[]
#-----------------for loop---------------

with open(RawfileName0,'r') as FILE:
    FILE.readline() # skip header line - skip row 1
    ## This reads the line and so does nothing with it
    for row in FILE:
        NextLabel,NextReview=row.split(",", 1)
        #print(Label)
        #print(Review)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel)
 ##----------------------------------------   
    
print(AllReviewsList)
print(AllLabelsList)

########################################
##
## CountVectorizer  and TfidfVectorizer
##
########################################
## Now we have what we need!
## We have a list of the contents (reviews)
## in the csv file.

My_CV1=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )

My_TF1=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )


## NOw I can vectorize using my list of complete paths to my files
X_CV1=My_CV1.fit_transform(AllReviewsList)
X_TF1=My_TF1.fit_transform(AllReviewsList)

print(My_CV1.vocabulary_)
print(My_TF1.vocabulary_)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
## The column names are the same for TF and CV
ColNames=My_TF1.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)
DataFrame_TF=pd.DataFrame(X_TF1.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CV = DataFrame_CV.rename(MyDict, axis = "index")
DataFrame_TF = DataFrame_TF.rename(MyDict, axis = "index")
DataFrame_CV.index.name = 'LABEL'
DataFrame_TF.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CV.columns)
#print(DataFrame_TF.columns)

## Let's build a small function that will find 
## numbers/digits and return True if so

##------------------------------------------------------
### DEFINE A FUNCTION that returns True if numbers
##  are in a string 
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in DataFrame_CV.columns:
    #print(nextcol)
    ## Remove unwanted columns
    Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)


    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
    
DataFrame_CV1 = DataFrame_CV.reset_index()
DataFrame_TF1 = DataFrame_TF.reset_index()

print(DataFrame_CV1)
print(DataFrame_TF1)

print(DataFrame_CV)
print(DataFrame_TF)

#DataFrame_CV1.to_csv(r'C:\Users/aivii\programsmm\HW3_736\cleaned_cv.csv', index = False)
#DataFrame_TF1.to_csv(r'C:\Users/aivii\programsmm\HW3_736\cleaned_tf.csv', index = False)





## Document 2

########################################################
##
##  GOAL 1: Read very dirty and labeled 
##          csv file (each row is a movie review
##          the labels are pos and neg - 
##          GET this data into a COPRUS and then
##          Use CountVectorizer to process it.
########################################################

# First way 
## Read in the file 
RawfileName="C:/Users/aivii/programsmm/HW3_736/MovieDataSAMPLE_labeledVERYSMALL.csv"
FILE=open(RawfileName,"r")  ## Don't forget to close this!

#################  Create a new Corpus Folder using os------
path="C:/Users/aivii/programsmm/HW3_736//SmallMovieCorpusMaya"
## You can only do this one!
IsFolderThere=os.path.isdir(path)
print(not(IsFolderThere)) # is the folder not there 

if (not(IsFolderThere)): # if is not there create it 
    MyNewCorpus=os.makedirs(path)



FILE.seek(0) # brigs to the BEGGINING to the file 

for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    #print(RawRow)
    
## OK - this works....so now we can loop through and create 
## .txt files from each row....

FILE.seek(0) # start again from the beggining 
counter=-1 # skip the first line 
for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    print(RawRow)
    
    ## In this case, the LABEL, you will notice, is at the END
    ##  of each row. 
    ## To get to the Label and use it to name each .txt file in the
    ## corpus, we will use SPLIT. In Python, using split on a string
    ## creates a list.
    
    ## First - before splitting - let's remove (strip) all
    ## the newlines, etc. 
    NextRow=row.strip()
    NextRow=NextRow.strip("\n") # strip new lines 
    NextRow=NextRow.rstrip(",") ## right strip comas
    
    MyList=NextRow.split(",")
    #print(MyList)
    
    # There are a lot of blanks that needs to be removed
    My_Blank_Filter = filter(lambda x: x != '', MyList) # anything that is not empty 
    ## Convert it back to a list - but without the blanks
    MyList = list(My_Blank_Filter) # add the filter to the new list 
    
    ## Let's look:
    #print("\n\nNEXT LIST \n:", MyList)
    
    TheLabel=MyList[-1]
    print(TheLabel)
    MyList.pop()
    ## Also - let's REMOVE this label from the list
    ## otherwise the label will be in the data
    ## and that is NOT PERMITTED when training and testing
    ## models
    
    
    counter=counter+1   ## this let's us know that we went
                ## through the loop at least once.
                
    if(counter>=1):
        ## If we are NOT in the first loop then we can
        ## start to build the .txt files.
        NewFileName=TheLabel+str(counter)+".txt"
        print(NewFileName)
        
        NewFilePath=path+"/"+NewFileName
        n_file=open(NewFilePath, "w")
        
        ## Now, we want to write the contents of MyList
        ## into a file. BUT - not as a list.
        
        MyNewString=" ".join(MyList)
        n_file.write(MyNewString)
        n_file.close()
    
    
FILE.close()


#################################################
##
## Now we have created a corpus and the names
## of the files in the corpus are the sentiment labels
## From here - we can use CountVectorizer to 
## Process this corpus into a DF
##
#############################################################

## Recall: This is where MY corpus is and what it is called
# C:\Users\aivii\programsmm\HW3_736\SmallMovieCorpusMaya

## From above, the variable "path" is the path to our new corpus

print(path)
print(os.listdir(path)) # print all the files in the path

## We will use CountVectorizer to format the corpus into a DF...
######## First - build the list of complete file paths
###--------------------------------------------------------

ListOfCompleteFilePathsMovies=[] # paths
ListOfJustFileNamesMovies=[]     # names

for filename in os.listdir(path):
    
    print(path+ "/" + filename)
    next=path+ "/" + filename
    
    next_file_name=filename.split(".")   
    nextname=next_file_name[0]
    ListOfCompleteFilePathsMovies.append(next)
    ListOfJustFileNamesMovies.append(nextname)

#print("DONE...")
print("full list...")
print(ListOfCompleteFilePathsMovies)
print(ListOfJustFileNamesMovies)

############# Now - use CountVectorizer.....................

MyVect5=CountVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## NOw I can vectorize using my list of complete paths to my files
X_Movies=MyVect5.fit_transform(ListOfCompleteFilePathsMovies)


## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesMovies=MyVect5.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_Movies=pd.DataFrame(X_Movies.toarray(),
                              columns=ColumnNamesMovies)
print(CorpusDF_Movies)

## Now update the row names
MyDictMovies={}
for i in range(0, len(ListOfJustFileNamesMovies)):
    MyDictMovies[i] = ListOfJustFileNamesMovies[i]

print("MY DICT:", MyDictMovies)
        
CorpusDF_Movies=CorpusDF_Movies.rename(MyDictMovies, axis="index")
print(CorpusDF_Movies)


############# Now - use TfidfVectorizer.....................

My_TF_Vect5=TfidfVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## NOw I can vectorize using my list of complete paths to my files
X_Movies_TF=My_TF_Vect5.fit_transform(ListOfCompleteFilePathsMovies)


## Let's get the feature names which ARE the words
ColumnNamesMoviesTF=My_TF_Vect5.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_MoviesTF=pd.DataFrame(X_Movies_TF.toarray(),
                              columns=ColumnNamesMoviesTF)
print(CorpusDF_MoviesTF)

## Now update the row names
MyDictMovies={}
for i in range(0, len(ListOfJustFileNamesMovies)):
    MyDictMovies[i] = ListOfJustFileNamesMovies[i]

print("MY DICT:", MyDictMovies)
        
CorpusDF_MoviesTF=CorpusDF_MoviesTF.rename(MyDictMovies, axis="index")
print(CorpusDF_MoviesTF)

## Row names
print(CorpusDF_MoviesTF.index)


def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------


for nextcol in CorpusDF_Movies.columns:
    #print(nextcol)
    ## Remove unwanted columns
    Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        CorpusDF_Movies=CorpusDF_Moviesdrop([nextcol], axis=1)
        CorpusDF_MoviesTF=CorpusDF_MoviesTF.drop([nextcol], axis=1)

   
    elif(len(str(nextcol))<=3):
        print(nextcol)
        CorpusDF_Movies=CorpusDF_Movies.drop([nextcol], axis=1)
        CorpusDF_MoviesTF=CorpusDF_MoviesTF.drop([nextcol], axis=1)

print(CorpusDF_Movies)
print(CorpusDF_MoviesTF)


CorpusDF_Movies = CorpusDF_Movies.rename(index={'neg3': 'neg', 'neg4': 'neg', 'neg5':'neg', 'pos1':'pos', 'pos2':'pos'})
print(CorpusDF_Movies)

CorpusDF_MoviesTF = CorpusDF_MoviesTF.rename(index={'neg3': 'neg', 'neg4': 'neg', 'neg5':'neg', 'pos1':'pos', 'pos2':'pos'})
print(CorpusDF_MoviesTF)

CorpusDF_Movies.index.name = 'LABEL'
CorpusDF_MoviesTF.index.name = 'LABEL'

CorpusDF_Movies1 = CorpusDF_Movies.reset_index()
print(CorpusDF_Movies1)

CorpusDF_MoviesTF1 = CorpusDF_MoviesTF.reset_index()
print(CorpusDF_MoviesTF1)


# Write to csv
CorpusDF_Movies1.to_csv(r'C:\Users/aivii\programsmm\HW3_736\movies_cleaned_cv.csv', index = False)
CorpusDF_MoviesTF1.to_csv(r'C:\Users/aivii\programsmm\HW3_736\movies_cleaned_tf.csv', index = False)


#######################################################################
## Document 2
# Second way 

## Read in the file 
filename = 'MovieDataSAMPLE_labeledVERYSMALL.csv'
file = open(filename, 'r') # 'r' is for 'read' permissions
file_data = [row for row in file]

## Prepare new clean file
clean_filename = "clean_movies.csv"
clean_file = open(clean_filename, 'w') # 'w' is for 'write' permissions
first_row = "label,text\n"

clean_file.write(first_row)
clean_file.close()

## Wrining out to a file 
output_filename = 'output.txt'
outfile = open(output_filename, 'w')
outfile.close()
outfile = open(output_filename, 'a')

def display_rows(file_data):
    for row in file_data:
        row = row.lstrip()
        row = row.rstrip()
        row = row.strip()
        raw_row = "\n\nROW:" + row + "\n"
        outfile.write(raw_row)
#         print(raw_row)

display_rows(file_data)
file.close()

output_filename = 'output.txt'
outfile = open(output_filename, 'w')
outfile.close()
outfile = open(output_filename, 'a')

def clean_word(word):
    word=word.lower()
    word=word.lstrip()
    word=word.lstrip("\\n")
    word=word.strip("\n")
    #word=word.strip("\\n")
    word=word.replace(",","")
    word=word.replace(" ","")
    word=word.replace("_","")
    word=re.sub('\+', ' ',word)
    word=re.sub('.*\+\n', '',word)
    word=re.sub('zz+', ' ',word)
    word=word.replace("\t","")
    word=word.replace(".","")
    #word=word.replace("\'s","")
    word=word.strip()
    word = word.replace("\\'","")
    if word not in ["", "\\", '"', "'", "*", ":", ";"]:
        if len(word) >= 3:
            if not re.search(r'\d', word): ##remove digits
                return word

clean_filename = "clean_movies2.txt"
cleanfile = open(clean_filename, 'w')
cleanfile.close()
cleanfile = open(clean_filename, 'a')

def display_rows(file_data):

    for row in file_data:
        
        row = row.lstrip()
        row = row.rstrip()
        row = row.strip()
        raw_row = "\n\nROW:" + row + "\n"
        #outfile.write(raw_row)
        row_list = row.split(" ")
        new_list = []
        for word in row_list:
             #to_put_in_outfile = "The next word BEFORE is: "+ word +"\n"
             #outfile.write(to_put_in_outfile)
            word = clean_word(word)
           #print(word)
            if word:
                new_list.append(word)
       #label = new_list[-1]
        label = ''.join(char for char in new_list[-1] if char.isalpha())
        new_list.pop()
        just_text = ' '.join(new_list)
        #print(label)
        #print(just_text)
        to_write = label + ',' + just_text + '\n'
        print('TO WRITE', to_write)
        cleanfile.write(to_write)

            
new_list = display_rows(file_data)
file.close()
cleanfile.close()


df = pd.read_csv('clean_movies_v2.txt', engine='python-fwf', sep=",")
print(df)

df['LABEL'] = df.apply(lambda x: x['textreviewclass'].split(',')[0], axis=1)
df['review'] = df.apply(lambda x: x['textreviewclass'].split(',')[1], axis=1)
df = df.drop('textreviewclass', axis=1)
print(df)

## Next step - use CV or tfidf

##################################################################################
## Document 2
# Third way 
dirtyFile = pd.read_csv('MovieDataSAMPLE_labeledVERYSMALL.csv')
df = pd.DataFrame()
df['review'] = dirtyFile[dirtyFile.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
print(df)

df['LABEL'] = df.apply(lambda x: x['review'][-3], axis=1)

def clean_rogue_characters(string):     
    exclude = ['\\',"\'"]
    string = '.'.join(string.split('\\n'))
    string = ''.join(ch for ch in string if ch not in exclude)
    return string 
df['review'] = df['review'].apply( lambda x: clean_rogue_characters(x) )

cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]

print(df)

df.to_csv(r'C:\Users\aivii\programsmm\HW3_736\movies_cleaned_MayaNew.csv', index = False)

RawfileName3="C:/Users/aivii/programsmm/HW3_736/movies_cleaned_MayaNew.csv"

## This file has a header. 
## It has "setinment" and "review" on the first row.

AllReviewsList=[]
AllLabelsList=[]
#-----------------for loop---------------

with open(RawfileName3,'r') as FILE:
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


My_CV3=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )

My_TF3=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )


## NOw I can vectorize using my list of complete paths to my files
X_CV3=My_CV3.fit_transform(AllReviewsList)
X_TF3=My_TF3.fit_transform(AllReviewsList)

print(My_CV3.vocabulary_)
print(My_TF3.vocabulary_)


ColNames=My_TF3.get_feature_names()


## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV3=pd.DataFrame(X_CV3.toarray(), columns=ColNames)
DataFrame_TF3=pd.DataFrame(X_TF3.toarray(), columns=ColNames)

# Update row names with file names
MyDict = {}
for i in range(0, len(AllLabelsList)):
    MyDict[i] = AllLabelsList[i]

print("MY DICT", MyDict)

DataFrame_CV3 = DataFrame_CV3.rename(MyDict, axis = "index")
DataFrame_TF3 = DataFrame_TF3.rename(MyDict, axis = "index")
DataFrame_CV3.index.name = 'LABEL'
DataFrame_TF3.index.name = 'LABEL'

## Drop/remove columns not wanted
print(DataFrame_CV3.columns)
#print(DataFrame_TF.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in DataFrame_CV3.columns:
  
    Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
 
    LogResult=Logical_Numbers_Present(nextcol)
    
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV3=DataFrame_CV3.drop([nextcol], axis=1)
        DataFrame_TF3=DataFrame_TF3.drop([nextcol], axis=1)


    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV3=DataFrame_CV3.drop([nextcol], axis=1)
        DataFrame_TF3=DataFrame_TF3.drop([nextcol], axis=1)
    
DataFrame_CV13 = DataFrame_CV3.reset_index()
DataFrame_TF13 = DataFrame_TF3.reset_index()

print(DataFrame_CV13)
print(DataFrame_TF13)

print(DataFrame_CV3)
print(DataFrame_TF3)

#DataFrame_CV3.to_csv(r'C:\Users/aivii\programsmm\HW3_736\cleaned_cv.csv', index = False)
#DataFrame_TF3.to_csv(r'C:\Users/aivii\programsmm\HW3_736\cleaned_tf.csv', index = False)

## Can split to test and train from here and model 
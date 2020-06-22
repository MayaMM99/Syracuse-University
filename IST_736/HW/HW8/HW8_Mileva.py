# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:20:41 2020

@author: Maya
HW8 Topic Modeling 
110th Congress
"""

## Load the packages
##################################################################
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import os
from nltk.corpus import stopwords
import pandas as pd
import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
#from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
#from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np


## Functions
######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

 ## implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
        
# Helper function
def plot_20_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Stemming 
STEMMER=PorterStemmer()
print(STEMMER.stem("singer"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


################################################################################################
# PART 1
## Use small text corpuses for example
# 3 texts-3 topics


all_file_names = []

path="C:\\Users\\aivii\\programsmm\\HW8_736\\Dog_Travel_Virus"
#print("calling os...")
#print(os.listdir(path))
FileNameList=os.listdir(path)
#print(FileNameList)
ListOfCompleteFiles=[]
for name in os.listdir(path):
    print(path+ "\\" + name)
    next=path+ "\\" + name
    ListOfCompleteFiles.append(next)
#print("DONE...")
print("full list...")
print(ListOfCompleteFiles)




english_stop_words = stopwords.words('english')
mine_stop_words = ['in', 'of', 'at', 'a', 'the', 'also', 'in', 'im']
extend_stop_words = english_stop_words+mine_stop_words 


MyVectLDA_DTV=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words= extend_stop_words,
                        max_features=100,
                        ##stop_words=["and", "or", "but"],
                        #token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        encoding = 'latin-1',
                        lowercase = True
                        )

#MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
Vect_DTV = MyVectLDA_DTV.fit_transform(ListOfCompleteFiles)


# Plot 
# Visualise the 10 most common words
plot_20_most_common_words(Vect_DTV, MyVectLDA_DTV)


ColumnNamesLDA_DTV=MyVectLDA_DTV.get_feature_names()
CorpusDF_DTV=pd.DataFrame(Vect_DTV.toarray(),columns=ColumnNamesLDA_DTV)

#print(ColumnNamesLDA_DTV)
print(CorpusDF_DTV)

num_topics = 3
lda_model_DTV = LatentDirichletAllocation(n_components=num_topics, max_iter=20, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DTV_Model = lda_model_DTV.fit_transform(Vect_DTV) # CAN DO A VECTORIZER OF THE DF!!!!!!!!!!!!!

print("SIZE: ", LDA_DTV_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in Dog, Travel, Virus data...")
print(LDA_DTV_Model[0])
print("Seventh Doc in Dog, Travel, Virus..")
print(LDA_DTV_Model[6])

## Print LDA using print function from above
print("LDA Dog, Travel, Virus Model:")
print_topics(lda_model_DTV, MyVectLDA_DTV)


####################################################
##
## VISUALIZATION
##
####################################################
"""import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model_DTV, Vect_DTV, MyVectLDA_DTV, mds='tsne')
### !!!!!!! Important - you must interrupt and close the kernet in Spyder to end
## In other words - press the red square and then close the small red x to close
## the Console
pyLDAvis.show(panel)"""

## Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model_DTV.components_)

## Assign Column and Index
df_topic_keywords.columns = MyVectLDA_DTV.get_feature_names()
topicnames = df_topic_keywords.index 

## View
df_topic_keywords.head()



## Another Vis 
word_topic = np.array(lda_model_DTV.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(ColumnNamesLDA_DTV)



fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

################################################################################################
# PART 2
## 110TH Congress 

# get the files 

all_file_names = []

path="C:\\Users\\aivii\\programsmm\\HW8_736\\COMBINED"
#print("calling os...")
#print(os.listdir(path))
FileNameList=os.listdir(path)
#print(FileNameList)
ListOfCompleteFiles=[]
for name in os.listdir(path):
    #print(path+ "\\" + name)
    next=path+ "\\" + name
    ListOfCompleteFiles.append(next)
#print("DONE...")
print("full list...")
#print(ListOfCompleteFiles)

english_stop_words = stopwords.words('english')
mine_stop_words = ['in', 'of', 'at', 'a', 'the', 'also', 'in', 'im', '2007','2008','act','american','chairman','committee','congress','country','doc',
                   'docno','don','floor','going','government','house','important','just','know','legislation','like','madam','make','members','mr',
                   'mrs','ms','need','new', 'people','president','representatives','say','speaker','state','states','support','text','thank',
                   'think','time', 'today','want','work','year', 'us', 'one', 'get', 'would', 'many', 'well', 'may', 'way', 'republican','republicans',
                   'democrats', 'democrat', 'take', 'let', 'good', 'bill', 'nea','res'

]

# Try with different again
additional_stop = [ '2007',
 '2008',
 'act',
 'american',
 'chairman',
 'committee',
 'congress',
 'country',
 'doc',
 'docno',
 'don',
 'floor',
 'going',
 'government',
 'house',
 'important',
 'just',
 'know',
 'legislation',
 'like',
 'madam',
 'make',
 'members',
 'mr',
 'mrs',
 'ms',
 'need',
 'new',
 'people',
 'president',
 'representatives',
 'say',
 'speaker',
 'state',
 'states',
 'support',
 'text',
 'thank',
 'think',
 'time',
 'today',
 'want',
 'work',
 'year', 'bill','also','would', 'one', 'many', 'well', 'would', ]

extend_stop_words = english_stop_words+additional_stop
regex = "[a-zA-Z]{3,15}"
MyVectLDA =CountVectorizer(input='filename',
                        #analyzer = 'word',
                        stop_words= extend_stop_words,
                        #max_features=20000,
                        ##stop_words=["and", "or", "but"],
                        token_pattern= regex,
                        #token_pattern=pattern,
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        #ngram_range = (1,2),
                        encoding = 'latin-1',
                        #lowercase = True
                        )

Vect = MyVectLDA.fit_transform(ListOfCompleteFiles)



# Plot 
# Visualise the 20 most common words
plot_20_most_common_words(Vect, MyVectLDA)



ColumnNamesLDA=MyVectLDA.get_feature_names()
CorpusDF=pd.DataFrame(Vect.toarray(),columns=ColumnNamesLDA)

## Drop/remove columns not wanted
#print(FinalDF_CV.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in CorpusDF.columns:
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
        CorpusDF=CorpusDF.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        CorpusDF=CorpusDF.drop([nextcol], axis=1)

#print(ColumnNamesLDA_DTV)
print(CorpusDF)
#CorpusDF.to_csv("C:\\Users\\aivii\\programsmm\\HW8_736\\COMBINED.csv")

## LDA
num_topics = 40
lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=20, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_Model = lda_model.fit_transform(Vect) # CAN DO A VECTORIZER OF THE DF!!!!!!!!!!!!!

print("SIZE: ", LDA_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in 100th congress data...")
print(LDA_Model[0])
print("Seventh Doc in 110th Congress data..")
print(LDA_Model[6])

## Print LDA using print function from above
print("LDA 110th Congress data  Model:")
print_topics(lda_model, MyVectLDA)


## Visualize 
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model, Vect, MyVectLDA, mds='tsne')
### !!!!!!! Important - you must interrupt and close the kernet in Spyder to end
## In other words - press the red square and then close the small red x to close
## the Console
pyLDAvis.show(panel)
pyLDAvis.save_html(panel, 'HW8_lda.html')


## Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)

## Assign Column and Index
df_topic_keywords.columns = MyVectLDA.get_feature_names()
topicnames = df_topic_keywords.index 

## View
df_topic_keywords.head()


## Another Vis 
word_topic = np.array(lda_model.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(ColumnNamesLDA)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 20

num_topics = 10
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.8)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()


###########################################################################
# Try another way
"""import os
def get_data_from_files(path):
    directory = os.listdir(path)
    results = []
    for file in directory:
        f=open(path+file,  encoding = "ISO-8859-1")
        results.append(f.read())
        f.close()
    return results

# DATA SET 2
data_fd = get_data_from_files('C:/Users/aivii/programsmm/HW8_736/110/110-f-d/')
data_fr = get_data_from_files('C:/Users/aivii/programsmm/HW8_736/110/110-f-r/')
data_md = get_data_from_files('C:/Users/aivii/programsmm/HW8_736/110/110-m-d/')
data_mr = get_data_from_files('C:/Users/aivii/programsmm/HW8_736/110/110-m-r/')

female_data = data_fd + data_fr 
male_data = data_md + data_mr
dem_data = data_md + data_fd
rep_data = data_mr + data_fr

all_data = female_data + male_data

## Cut it in half
# DATA SET 2 -- SMALL
female_data_sm = data_fd[:50] + data_fr[:50] 
male_data_sm = data_md[:50] + data_mr[:50]


all_data_sm = female_data_sm + male_data_sm

regex = "[a-zA-Z]{3,15}"
english_stop_words = stopwords.words('english')
mine_stop_words = ['in', 'of', 'at', 'a', 'the', 'also', 'in', 'im', '2007','2008','act','american','chairman','committee','congress','country','doc',
                   'docno','don','floor','going','government','house','important','just','know','legislation','like','madam','make','members','mr',
                   'mrs','ms','need','new', 'people','president','representatives','say','speaker','state','states','support','text','thank',
                   'think','time', 'today','want','work','year', 'us', 'one', 'get', 'would', 'many', 'well', 'may', 'way']
extend_stop_words = english_stop_words+mine_stop_words 

MyVectLDA1=CountVectorizer(#input='filename',
                        #analyzer = 'word',
                        stop_words= extend_stop_words,
                        #max_features=90000,
                        ##stop_words=["and", "or", "but"],
                        token_pattern= regex,
                        #token_pattern=pattern,
                        #tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode',
                        #ngram_range = (1,2),
                        #encoding = 'latin-1',
                        #lowercase = True
                        )

Vect1 = MyVectLDA1.fit_transform(all_data_sm)

# Plot 
# Visualise the 20 most common words
plot_20_most_common_words(Vect1, MyVectLDA1)



ColumnNamesLDA1=MyVectLDA1.get_feature_names()
CorpusDF1=pd.DataFrame(Vect1.toarray(),columns=ColumnNamesLDA1)

## Drop/remove columns not wanted
#print(FinalDF_CV.columns)

def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)


for nextcol in CorpusDF1.columns:
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
        CorpusDF1=CorpusDF1.drop([nextcol], axis=1)
       
    elif(len(str(nextcol))<=3):
        print(nextcol)
        CorpusDF1=CorpusDF1.drop([nextcol], axis=1)

#print(ColumnNamesLDA_DTV)
print(CorpusDF1)

## LDA
num_topics = 40
lda_model1 = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_Model1 = lda_model1.fit_transform(Vect1) # CAN DO A VECTORIZER OF THE DF!!!!!!!!!!!!!

print("SIZE: ", LDA_Model1.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Doc in 100th congress data...")
print(LDA_Model1[0])
print("Seventh Doc in 110th Congress data..")
print(LDA_Model1[6])

## Print LDA using print function from above
print("LDA 110th Congress data  Model:")
print_topics(lda_model1, MyVectLDA1)


## Visualize 
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model1, Vect1, MyVectLDA1, mds='tsne')
### !!!!!!! Important - you must interrupt and close the kernet in Spyder to end
## In other words - press the red square and then close the small red x to close
## the Console
pyLDAvis.show(panel)
pyLDAvis.save_html(panel, 'HW8_lda1.html')


## Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model1.components_)

## Assign Column and Index
df_topic_keywords.columns = MyVectLDA1.get_feature_names()
topicnames = df_topic_keywords.index 

## View
df_topic_keywords.head()


## Another Vis 
word_topic = np.array(lda_model1.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(ColumnNamesLDA1)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()"""
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:28:44 2020

@author: Maya
"""

import nltk 
from nltk import FreqDist
import re
from textblob import TextBlob
from IPython.display import display, HTML
import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import * 



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

# Vader 
sid = SentimentIntensityAnalyzer()

# 1. Movie reviews 
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

print(len(neg_tw))
print(len(pos_tw))
print(len(neg_mv))
print(len(pos_mv))

# TextBlob
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
 
# twwets
display(pd.DataFrame(get_sentiment(neg_tw, 'neg')))
display(pd.DataFrame(get_sentiment(pos_tw, 'pos')))

# movies
display(pd.DataFrame(get_sentiment(neg_mv, 'neg')))
display(pd.DataFrame(get_sentiment(pos_mv, 'pos')))


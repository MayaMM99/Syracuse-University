# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:34:08 2020

@author: Maya
IST 736, HW5 

Training/Evaluating Data Aquisition Through AMT
"""

## EXPERIMENT 1

# Load the data 

import pandas as pd
import numpy as np

neg = pd.read_csv('amt_neg.csv')
pos = pd.read_csv('amt_pos.csv')

print(neg.head(3))
print(pos.head(3))

# Check the colum names
print(neg.columns.tolist())

# Find how many people worked
def get_unique(df, column):
    unique = np.unique(df[column], return_counts=True)
    df = pd.DataFrame(zip(unique[0], unique[1]))
    return len(unique[0]), unique, df

num_neg, unique_neg, u_neg_df = get_unique(neg, 'WorkerId')    
num_pos, unique_pos, u_pos_df = get_unique(pos, 'WorkerId')

print(num_neg, 'Turkers worked on neg tweets:')
print(num_pos, 'Turkers worked on pos tweets: ')

# Negative hits
u_neg_df.plot(kind='bar',x=0,y=1, color = '#4a4e69')

# Positive hits
u_pos_df.plot(kind='bar',x=0,y=1, color = '#4a4e69')

# min and max 
print('For {}, the min was: {} and the max was: {}'.format('neg', unique_neg[1].min(), unique_neg[1].max())) 
print('For {}, the min was: {} and the max was: {}'.format('pos', unique_pos[1].min(), unique_pos[1].max())) 

# how long 
# neg
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette("pastel")
sns.catplot(x="Answer.sentiment.label", 
            y="WorkTimeInSeconds", 
            kind="bar", 
            order=['Negative', 'Neutral', 'Positive'], 
            data=neg);

plt.title('Negative')

# pos
sns.catplot(x="Answer.sentiment.label", 
            y="WorkTimeInSeconds", 
            kind="bar", 
            order=['Negative', 'Neutral', 'Positive'], 
            data=pos)
plt.title('Positive')

# Check the response time 
response_time = neg[neg['WorkTimeInSeconds'] < 10]
response_time_check = neg[neg['WorkTimeInSeconds'] > 10]


# cheack for boths

# low response time
count = pos.groupby(['WorkerId'])['HITId'].count()
work_time = pos.groupby(['WorkerId'])['WorkTimeInSeconds'].mean()
new_df = pd.DataFrame([work_time, count]).T
new_df.reset_index(inplace=True)

df = new_df.copy()
df = df[['WorkerId', 'WorkTimeInSeconds', 'HITId']]
print(df[:5])

# high response time 
new_df['WorkTimeInMin'] = new_df['WorkTimeInSeconds']/60

df = new_df.copy()
df = df.sort_values(by='WorkTimeInMin', ascending=False)
df = df[['WorkerId', 'WorkTimeInMin', 'HITId']]
print(df[:5])

# count the answers
count = pd.DataFrame(pos.groupby(['WorkerId', 'Answer.sentiment.label'])['HITId'].count())

df = count.copy()
print(df[:10])

# Check the answers 
pnn = pd.DataFrame()
pnn['Neutral'] = pos.groupby('WorkerId')['Answer.sentiment.label'].apply(lambda x: (x=='Neutral').sum())
pnn['Positive'] = pos.groupby('WorkerId')['Answer.sentiment.label'].apply(lambda x: (x=='Positive').sum())
pnn['Negative'] = pos.groupby('WorkerId')['Answer.sentiment.label'].apply(lambda x: (x=='Negative').sum())
pnn['Total'] = pos.groupby('WorkerId')['Answer.sentiment.label'].apply(lambda x: x.count())

df = pnn.copy()
print(df[:10])

top = pnn.sort_values(by=['Total'], ascending=False)
df = top.copy()
print(df[:10])

# look at the response time 
top['Avg_WorkTimeInSeconds'] = pos.groupby('WorkerId')['WorkTimeInSeconds'].apply(lambda x: x.mean())
top['Avg_WorkTimeInMin'] = pos.groupby('WorkerId')['WorkTimeInSeconds'].apply(lambda x: x.mean()/60)
top['Min_WorkTimeInMin'] = pos.groupby('WorkerId')['WorkTimeInSeconds'].apply(lambda x: x.min()/60)
top['Max_WorkTimeInMin'] = pos.groupby('WorkerId')['WorkTimeInSeconds'].apply(lambda x: x.max()/60)
df = top.copy()
df.reset_index(inplace=True)
df = df[['WorkerId', 'Neutral', 'Positive','Negative','Avg_WorkTimeInMin']]

print(df[:10])

#######################################################################

## EXPERIMENT 2 - only master turkers

# Load the data 
df2 = pd.read_csv('amt_all.csv')
print(df2.head())
len(df2)


df3 = df2.copy()
#print(df3.head(4))
df3.reset_index(inplace=True)
df3 = df3[['WorkerId', 'Answer.sentiment.label']]
print(df3[:10])

# have to get the labels
# Getting labels...
labels = pd.read_csv('all_labeled.csv')

labels = labels.append([labels] * 2, ignore_index=True)
print(len(labels))
df = labels.copy()
df['short'] = df.apply(lambda x: x['0'].split(' ')[:5], axis=1)

df = df[['PoN', 'short']]
#print(df)
print(df[:10]) # good


# sort them so they match 
sorted_labels = labels.sort_values(by=['0'])
#print(sorted_labels)
sorted_df2 = df2.sort_values(by=['Input.text'])

sorted_df2['Input.text'][:5]# they are sorted now 
#print(sorted_labels.head(30))

#print(sorted_df2['Input.text'].head(30))

# They matched 

sorted_df2['PoN'] = sorted_labels['PoN'].tolist() # get them 
#print(sorted_df2.head(30))
df = sorted_df2[sorted_df2.columns[-5:]][:10]

df['short'] = df.apply(lambda x: x['Input.text'].split(' ')[1:3], axis=1)


df = df[['short', 'Answer.sentiment.label', 'PoN']]
print(df[:10]) # guess they did a good job 

# clean it 
all_df = sorted_df2[['Input.text', 'WorkerId', 'Answer.sentiment.label', 'PoN']]

df = all_df.copy()
df = df[['WorkerId', 'Answer.sentiment.label', 'PoN']]
print(df[:10])

all_df_all = all_df.copy()
all_df_all['APoN'] = all_df_all.apply(lambda x: x['Answer.sentiment.label'][0], axis=1)

all_df_all['agree'] = all_df_all.apply(lambda x: x['PoN'] == x['APoN'], axis=1)

df = all_df_all[-10:].copy()
df = df[['WorkerId', 'PoN', 'APoN', 'agree']]
print(df[:10])

result_df = pd.DataFrame(all_df_all.groupby(['Input.text','PoN'])['agree'].mean())
result_df = result_df.reset_index()
df = result_df.copy()
df = df[['PoN', 'agree']]
#print(df[:10])

# do it with function 
def return_agreement(num):
    if num == 0:
        return 'agree_wrong'
    if num == 1:
        return 'agree'
    if (num/1) !=0:
        return 'disparity'

result_df['agree_factor'] = result_df.apply(lambda x: return_agreement(x['agree']), axis=1)
result_df

df = result_df.copy()
df = df[['PoN', 'agree', 'agree_factor']]
print(df[:10])

df1 = result_df.groupby(['agree_factor']).count()
df1.reset_index(inplace=True)
df = df1.copy()
df = df[['agree_factor','Input.text','PoN', 'agree']]
df1 = result_df.groupby(['agree_factor']).count()
df1.reset_index(inplace=True)
df = df1.copy()
df = df[['agree_factor','Input.text','PoN', 'agree']]
# print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

sns.barplot(x="agree_factor", y="agree", data=df1);

df2 = result_df.groupby(['agree_factor', 'PoN']).count()
df2.reset_index(inplace=True)

sns.barplot(x="agree_factor",
           y="agree",
           hue="PoN",
           data=df2);
plt.title("Pos/Neg split")

### EXPERIMENT 3 - one file again 
# include Cohen Kappa

# Load the last file 
df_3 = pd.read_csv('amt_all21.csv')
print(df_3[:5])
print(len(df_3))

print(df3.head(4))
from tabulate import tabulate # make it nice 
print(tabulate(df3[:10], tablefmt="rst", headers=df3.columns))

# Getting labels...
labels = pd.read_csv('all_labeled.csv')

labels = labels.append([labels] * 4, ignore_index=True)
print(len(labels))
df = labels.copy()
df['short'] = df.apply(lambda x: x['0'].split(' ')[:5], axis=1)

df = df[['PoN', 'short']]
#print(df)
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

# again sort them on tweets
sorted_labels = labels.sort_values(by=['0'])
sorted_turker = df_3.sort_values(by=['Input.text'])

sorted_turker['PoN'] = sorted_labels['PoN'].tolist()
df = sorted_turker[sorted_turker.columns[-5:]][:10]

df['short'] = df.apply(lambda x: x['Input.text'].split(' ')[1:3], axis=1)


df = df[['short', 'Answer.sentiment.label', 'PoN']]
#print(df[:10])
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

# clean the df
all_df = sorted_turker[['Input.text', 'WorkerId', 'Answer.sentiment.label', 'PoN']]

df = all_df.copy()
df = df[['WorkerId', 'Answer.sentiment.label', 'PoN']]
print(df[:10])
#print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

all_df_all = all_df.copy()
all_df_all['APoN'] = all_df_all.apply(lambda x: x['Answer.sentiment.label'][0], axis=1)

all_df_all['agree'] = all_df_all.apply(lambda x: x['PoN'] == x['APoN'], axis=1)


df = all_df_all[-10:].copy()
df = df[['WorkerId', 'PoN', 'APoN', 'agree']]
#print(tabulate(df[:10], tablefmt="rst", headers=df.columns))
print(df[:10])


# how many agreed
agree_df = pd.DataFrame(all_df_all.groupby(['Input.text','PoN'])['agree'].mean())
agree_df = agree_df.reset_index()
df = agree_df.copy()
df = df[['PoN', 'agree']]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

def return_agreement(num):
    if num == 0:
        return 'agree_wrong'
    if num == 1:
        return 'agree'
    if (num/1) !=0:
        return 'disparity'

agree_df['agree_factor'] = agree_df.apply(lambda x: return_agreement(x['agree']), axis=1)
agree_df

df = agree_df.copy()
df = df[['PoN', 'agree', 'agree_factor']]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))


df1 = agree_df.groupby(['agree_factor']).count()
df1.reset_index(inplace=True)
df = df1.copy()
df = df[['agree_factor','Input.text','PoN', 'agree']]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

sns.barplot(x=['Agreed', 'Disagreed'],
           y= [64,34],
           data = df1);
plt.title('How many turkers agreed on sentiment?')

sns.barplot(x="agree_factor", y="agree", data=df1);
plt.title('How many turkers agreed on sentiment, but were wrong?')

df2 = agree_df.groupby(['agree_factor', 'PoN']).count()
df2.reset_index(inplace=True)

sns.barplot(x="agree_factor",
           y="agree",
           hue="PoN",
           data=df2);
plt.title("Pos/Neg split")

### Kappa 

# get the columns needed 
turker_clean = df_3[['ReviewID', 'WorkerId', 'Answer.sentiment.label', 'Input.text']]

turker_clean

df = turker_clean.copy()
df = df[['ReviewID','WorkerId', 'Answer.sentiment.label']]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

turker_counts = pd.DataFrame(turker_clean.WorkerId.value_counts())

df = turker_counts.copy()
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

# pick top 5
t1 = turker_clean[turker_clean['WorkerId'] == 'A2RKUDGK5PQ44X']
t2 = turker_clean[turker_clean['WorkerId'] == 'A2L746JBCNW066']
t3 = turker_clean[turker_clean['WorkerId'] == 'A3F9N2P4NUUR7S']
t4 = turker_clean[turker_clean['WorkerId'] == 'A746183X8258H']
t5 = turker_clean[turker_clean['WorkerId'] == 'A3UF6XXFFRR237']

t1.reset_index(drop=True, inplace=True)
t2.reset_index(drop=True, inplace=True)
t3.reset_index(drop=True, inplace=True)
t4.reset_index(drop=True, inplace=True)
t5.reset_index(drop=True, inplace=True)

# combine them together
merged_df = pd.concat([t1, t2, t3, t4, t5], axis=0, sort=False)
merged_df.reset_index(drop=True, inplace=True)
print(merged_df)#they all got different number

df = merged_df.sort_values(by='WorkerId')
df = df[['WorkerId', 'Answer.sentiment.label']]
print(tabulate(df[:20], tablefmt="rst", headers=df.columns))

merged_df2 = pd.concat([t1, t2], axis=0, sort=False)


df = pd.DataFrame({'Turker': merged_df['WorkerId'].tolist(),
                   'SENTIMENT': merged_df['Answer.sentiment.label'].tolist(),
                   'REVIEW': merged_df['ReviewID'].tolist() })

grouped = df.groupby('Turker')
values = grouped['REVIEW'].agg('sum')
id_df = grouped['SENTIMENT'].apply(lambda x: pd.Series(x.values)).unstack()
id_df = id_df.rename(columns={i: 'SENTIMENT{}'.format(i + 1) for i in range(id_df.shape[1])})
result = pd.concat([id_df, values], axis=1)
result_df = pd.DataFrame(result)

df = result_df.T.copy()
df = df[df.columns[1:4]]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

t1 = result_df.T['A2RKUDGK5PQ44X'].tolist()
t2 = result_df.T['A3F9N2P4NUUR7S'].tolist()
t3 = result_df.T['A3UF6XXFFRR237'].tolist()
t4 = result_df.T['A2L746JBCNW066'].tolist()

#t1 = t1[:-1][:5]
#t2 = t2[:-1][:5]

print(t1[:-1][:5]) #thats all we can compare
print(t2[:-1][:5])
print(t3[:5])
print(t4[:5])

from sklearn.metrics import cohen_kappa_score
y1 = t1[:-1]
y2 = t2[:-1]
print(cohen_kappa_score(y1,y2))

y2 = t2[:-1]
y3 = t3[:-1]
print(cohen_kappa_score(y2,y3))

y1 = t1[:-1]
y3 = t3[:-1]
print(cohen_kappa_score(y1,y3))

y2 = t2[:-1]
y4 = t4[:-1]
print(cohen_kappa_score(y2,y4))


#### Another way 
new_turker_ids = pd.factorize(turker_clean['WorkerId'].tolist())

t_ids = ['T_' + str(id) for id in new_turker_ids[0]]

print(t_ids[:5])

turker_clean['T_ID'] = t_ids
print(turker_clean[:5])

turker_clean['sentiment'] = turker_clean.apply(lambda x: x['Answer.sentiment.label'][0], axis=1)
print(turker_clean[:5])

#clean = clean.reindex(columns=['ReviewID','T_ID', 'sentiment'])
last_df = turker_clean[['ReviewID', 'T_ID', 'sentiment']]
print(last_df.head(30))

df = last_df[:5]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

def get_array_of_reviews(turker, df):
    a = ['nan']*98
    df = last_df[last_df['T_ID'] == turker] 
    t_reviews = df['ReviewID'].tolist()
    t_sentiment = df['sentiment'].tolist()
    for index,review in enumerate(t_reviews):
        a[review] = t_sentiment[index]
#     print(t_reviews)

    return a

sparse_df = last_df.copy()
sparse_df['big_array'] = sparse_df.apply(lambda x: get_array_of_reviews(x['T_ID'], last_df), axis=1)
t0 = last_df[last_df['T_ID'] == 'T_0']

# look at the one
df = t0
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

# make mattrix so is easier to see
sparse_df['big_array_sm'] = sparse_df.apply(lambda x: x['big_array'][:5], axis=1)
df = sparse_df[['ReviewID', 'T_ID','sentiment', 'big_array_sm']]
print(tabulate(df[:10], tablefmt="rst", headers=df.columns))

# check if the resluts are correct 
#t0 = sparse_df[sparse_df['T_ID'] == 'T_0']

y1 = sparse_df['big_array'][sparse_df['T_ID'] == 'T_0'].tolist()[0]
y2 = sparse_df['big_array'][sparse_df['T_ID'] == 'T_1'].tolist()[0]
print(cohen_kappa_score(y1,y2))

# do it with a function
def calculate_kappa(num):
    y1 = sparse_df['big_array'][sparse_df['T_ID'] == 'T_'+str(num)].tolist()[0]
    y2 = sparse_df['big_array'][sparse_df['T_ID'] == 'T_'+str(num + 1)].tolist()[0]
    return cohen_kappa_score(y1,y2)

kappas = [calculate_kappa(num) for num in range(16)]

print(kappas)
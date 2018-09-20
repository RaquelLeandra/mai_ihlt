
# coding: utf-8

# # Mandatory exercise

# In[15]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import urllib.request
from nltk.metrics import jaccard_distance
from scipy.stats import pearsonr


# **Read all pairs of sentences of the trial set within the
# evaluation framework of the project.**

# In[2]:


pairs = {}

with open('trial/STS.input.txt', 'r') as f:
    for l in f:
        sid = l.split('\t')[0]
        s1  = l.split('\t')[1]
        s2  = l.split('\t')[2][:-1] # Remove the \n character
        pairs[sid] = [s1, s2]
        print(sid, pairs[sid])


# **Compute their similarities by considering words and Jaccard distance in lemmas.**
# 

# In[74]:


simmilarities = []

wnl = WordNetLemmatizer()

def lemmatize(sentence):
    words = nltk.word_tokenize(sentence)
    pairs = pos_tag(words)
    lemms = []
    for pair in pairs:
        if pair[1][0] in {'N', 'V'}:
            lemms.append(wnl.lemmatize(pair[0].lower(), pos=pair[1][0].lower()))
        else:
            lemms.append(pair[0].lower())
    return lemms

for sid in pairs:
    s1      = pairs[sid][0]
    s2      = pairs[sid][1]
    lemms_1 = set(lemmatize(s1))
    lemms_2 = set(lemmatize(s2))
    jaccard_simmilarity = 1 - jaccard_distance(lemms_1, lemms_2)
    simmilarities.append(jaccard_simmilarity) 
    print('id:', sid, 'similarity [0-1]:', jaccard_simmilarity)


# **Compare the previous results with gold standard by giving the pearson correlation between them.**

# In[75]:


gs = {}

with open('trial/STS.gs.txt', 'r') as f:
    for l in f:
        sid     = l.split('\t')[0]
        value   = abs( int(l.split('\t')[1])-5)    
        gs[sid] = value

refs = list(gs.values())
print(pearsonr(refs, simmilarities)[0])


# We obtain a correlation coefficient of 0.5790860088205633. 
# **This correlation coefficiente is better than the one in the previous exersice. This meaning that working with lemmas is better for this example.**
# 
# Working with lemmas should be generally better because the same words implying the same meaning may appear in a pair of sentences, but if they have any morphological variation they will be trated as separated words in the Jaccard Distance calculation.
# 
# _Also, as in the previus mandatory exercise:_
# 
# > As we are comparing two arrays of similarity values, we obtain a positive correlation. 
# 
# > This value is a little bigger than 0.5, this means that there is little correlation between the two arrays, so probably the Jaccard distance isn't the best way to measure the semantic similarity between sentences. 
# 
# > These results are due to the definition of Jaccard distance. This definition is fully based on set theory and does not take into account the semantic relationship between words (like synonymity).

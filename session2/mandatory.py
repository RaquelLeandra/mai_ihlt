
# coding: utf-8

# # Mandatory exercise

# In[1]:


import nltk
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


# **Compute their similarities by considering words and Jaccard distance.**
# 

# In[5]:


simmilarities = []
for sid in pairs:
    s1      = pairs[sid][0]
    s2      = pairs[sid][1]
    words_1 = set([word.lower() for word in nltk.word_tokenize(s1)])
    words_2 = set([word.lower() for word in nltk.word_tokenize(s2)])
    jaccard_simmilarity = 1 - jaccard_distance(words_1, words_2)
    simmilarities.append(jaccard_simmilarity) 
    print('id:', sid, 'similarity [0-1]:', jaccard_simmilarity)


# **Compare the previous results with gold standard by giving the pearson correlation between them.**

# In[4]:


gs = {}

with open('trial/STS.gs.txt', 'r') as f:
    for l in f:
        sid     = l.split('\t')[0]
        value   = abs( int(l.split('\t')[1])-5)    
        gs[sid] = value

refs = list(gs.values())
print(pearsonr(refs, simmilarities)[0])


# We obtain a correlation coefficient of 0.5140573923420935. 
# 
# As we are comparing two arrays of similarity values, we obtain a positive correlation. 
# 
# This value is a little bigger than 0.5, this means that there is little correlation between the two arrays, so probably the Jaccard distance isn't the best way to measure the semantic similarity between sentences. 
# 
# These results are due to the definition of Jaccard distance. This definition is fully based on set theory and does not take into account the semantic relationship between words (like synonymity).


# coding: utf-8

# # Optional exercise
# Build a language classifier

# In[1]:


import re
import collections
import time
import math
import multiprocessing
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics.scores import accuracy
from nltk.metrics import ConfusionMatrix
from time import process_time


# The next functions is used just to *pretty* display the elapsed time

# In[2]:


def showTime(seconds):
    if type(seconds) is float:
        seconds = round(seconds * 1000)
        return showTime(seconds // 1000) + str(seconds % 1000) + 'ms'
    if   seconds < 59:    return str(seconds)          + 's '
    elif seconds < 3599:  return str(seconds // 60)    + 'm ' + showTime(seconds % 60)
    elif seconds < 86399: return str(seconds // 3600)  + 'h ' + showTime(seconds % 3600)
    else:                 return str(seconds // 86400) + 'd ' + showTime(seconds % 86400)


# ### Trigrams class 
# Describes a set of trigrams with their names and frequencies
# * all: dictionary with trigrams frequencies as values and names as keys
# * min: minimum frequency to keep a trigram
# * B:   total number of different trigrams
# * N:   total number of trigrams

# In[3]:


class Trigrams:
    def __init__(self, minimum=0):  # minimum frecuency to keep trigram
        self.all = collections.defaultdict(lambda:0) # Default values for new keys is 0
        self.min = minimum
        self.B   = 0
        self.N   = 0
    
    def build(self, txt):
        self.all.clear()
        finder = TrigramCollocationFinder.from_words(txt)
        finder.apply_freq_filter(self.min)
        for name, freq in finder.ngram_fd.items():
            self.all[''.join(name)] = freq
            self.N += freq
        self.N = sum(self.all.values())
        self.B = len(self.all)
            
    #### Without nltk
    def build_NoNLTK(self, txt):
        size = 3 # Trigrams
        self.all.clear()
        for i in range(len(txt) - size):
            name = ''.join(txt[i + c] for c in range(self.n))
            self.all[name] += 1
        for name, freq in list(self.ngrams.items()):
            if freq < self.min:
                del self.all[name]
        self.N = sum(self.all.values())
        self.B = len(self.all)


# ### Text class
# Parent class containing the function to preprocess text lines

# In[4]:


class Text:
    # Removes all digits and punctuations
    # Remove spaces at the start of the sentence
    # Convert to lower case
    # Convert multiple spaces to single ones
    def preprocess(self, line):
        line = re.sub(r'(\d|[^\w ])', ' ', line)
        line = re.sub(r'^[ ]+', '', line)
        line = line.lower()
        line = re.sub(r'[ ]+', ' ', line)
        return line


# ### Lang class
# Child class of *Text* representing a language model with a *Trigrams set*
# * name: language name
# * trigrams: model trigrams

# In[5]:


class Lang(Text):
    def __init__(self, name):
        super()
        self.name     = name
        self.trigrams = Trigrams(minimum=5) # Minimum frequency to keep trigram = 5
        
    def buildModel(self, lines):
        txt = ''
        for line in lines:
            txt  += self.preprocess(line).strip() + '  ' # Join lines within two spaces
        self.trigrams.build(txt)


# ## Generate Language models
# Each language model will be built from a list of 30,000 sentences preprocessed and using character trigrams

# In[6]:


names = ['deu', 'eng', 'fra', 'ita', 'nld', 'spa'] # All languages to use and test
langs = {} # An entry for each language model


# In[7]:


print('Generating lang models, please wait...')
start = process_time()

for name in names:
    langs[name] = Lang(name)
    try:
        with open('langId/%s_trn.txt' % name, 'r', encoding='utf8') as f: langs[name].buildModel(f.readlines())
        print('Model for language "%s" complete.' % name)
    except FileNotFoundError:
        print('Files for language "%s" not found.' % name)

print('Done in %s' % showTime(process_time() - start))


# ## Generate tests sets

# ### Sentence class
# Each sentence will contain:
# * txt: original text
# * guessLabel: how the text is classified 
# * realLabel: how the text should have been classified
# * trigram: preprocessed text divided in trigrams and frequencies
# * probs: probability of being classified as each language
# 

# In[8]:


class Sentence(Text):
    def __init__(self, txt, realLabel='', guessLabel=''):
        self.txt        = txt
        self.guessLabel = guessLabel
        self.realLabel  = realLabel
        self.trigrams   = Trigrams()
        self.probs      = collections.defaultdict(lambda:0) # Default probability for new keys is 0
        self.trigrams.build(self.preprocess(self.txt))


# ### Test class
# Holds a list of *Sentences* to be classified
# * sentences: -
# * l: lambda value for the Laplace smooth technique
# * accuracy: accuracy of all the classification process
# * cmatrix: confusion matrix of all the classification process

# In[9]:


class Test(Text):
            
    def __init__(self): 
        super()
        self.sentences = []
        self.l         = 1
        self.accuracy  = 0
        self.cmatrix   = None
    
    # Add a sentece to be tested
    def addSentences(self, lines, realLabel):
        for line in lines:
            self.sentences.append(Sentence(line, realLabel))
            
    # Execute the test in all the sentences
    def execute(self, langs):
        ref = [] # Reference (real classification)
        clf = [] # Classified
        for sentence in self.sentences:
            self.classify(sentence, langs) # Classification
            ref.append(sentence.realLabel)
            clf.append(sentence.guessLabel)
        self.accuracy = accuracy(ref, clf)
        self.cmatrix  = ConfusionMatrix(ref, clf)
    
    # Classify a sentence in a language
    def classify(self, sentence, langs):
        best = float('-inf') # Best classification score
        
        for lang in langs:
            prob = 0
            for name, freq in sentence.trigrams.all.items():
                Ct = lang.trigrams.all[name]
                Nt = lang.trigrams.N
                B  = lang.trigrams.B
                prob += math.log((Ct + self.l) / (Nt + self.l * B)) * freq # Laplace smooth technique
            if prob > best: # Check if this language fits better
                best = prob
                sentence.guessLabel = lang.name
            sentence.probs[lang.name] = prob
                
        


# First, tests sentences are loaded, 10,000 per language

# In[10]:


test = Test()

print('Generating tests, please wait...')
start = process_time()

for name in names:
    try:
        with open('langId/%s_tst.txt' % name, 'r', encoding='utf8') as f: test.addSentences(f.readlines(), name)
        print('Loaded sentences for language "%s" test complete.' % name)
    except FileNotFoundError:
        print('Files for language "%s" not found.' % name)

print('Done in %s' % showTime(process_time() - start))


# Then, we run the tests...

# In[11]:


print('Executing tests, please wait...')
start = process_time()

test.execute(langs.values())
print('Accuracy:', test.accuracy, '\n')
print(test.cmatrix.pretty_format())

print('Done in %s' % showTime(process_time() - start))


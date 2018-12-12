from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import pandas as pd
import nltk
import re

from stop_words import get_stop_words
from nltk.corpus import stopwords


class Preprocessor:
    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        self.tagger = PerceptronTagger()
        self.lemmatizer = WordNetLemmatizer()

    def run(self, data):
        copy = data.copy()
        self.run_cleaner(copy)
        self.run_meaning(copy)
        self.as_string(copy)
        return copy

    def run_cleaner(self, data):
        for column in data.columns:
            data[column] = data[column].apply(word_tokenize)
            # data[column] = data[column].apply(self.auto_correct)

    def run_meaning(self, data):
        for column in data.columns:
            data[column] = data[column].apply(self.remove_stop_words)
            # data[column] = data[column].apply(self.tagger.tag)
            # data[column] = data[column].apply(self.lemmatize)
            # data[column] = data[column].apply(self.meaning)
            # data[column] = data[column].apply(self.revectorize)
            # Join together names ?
            # Remove stopwords ?

    def as_string(self, data):
        for column in data.columns:
            data[column] = data[column].str.join(' ')

    def remove_stop_words(self, vector):
        new_vector = []
        for word in vector:
            #word = re.sub(r'\W+', '', word)
            #print(word)
            #if not word in ['\'s'] and not word in list(stopwords.words('english')):
            #if len(word) > 0 and not word in list(stopwords.words('english')):
            if not word in list(stopwords.words('english')):
                #print(word)
                new_vector.append(word)
        return new_vector

    def revectorize(self, tagged):
        return [word for word, tag in tagged]

    def auto_correct(self, vector):
        return [spell(word) for word in vector]

    def lemmatize(self, tagged):
        return [(self.lemmatizer.lemmatize(word), tag) for word, tag in tagged]


    def meaning(self, tagged):

        morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                      'VB': wn.VERB, 'RB': wn.ADV,
                      'NNS': wn.NOUN}

        #print(tagged)
        semantic = []
        for idx, (token, tag) in enumerate(tagged):  # For each word
            semantic.append((token, tag))
            if tag in ['NN', 'NNS', 'VB', 'JJ', 'RB']:
                context = [i for i,_ in tagged if i != token]  # Context for the word ({S} - {word})
                #synset = lesk(context, token.lower(), 'n')  # Lesk algorithm => Synset
                synset = wn.synsets(token, morphy_tag[tag])
                if synset:
                    synset = re.sub(r'Synset\(\'(.*)\..*\..*', r'\1', str(synset[0]))
                    semantic[-1] = (synset, tag)

        #print(semantic)
        return semantic

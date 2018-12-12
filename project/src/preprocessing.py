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

class Preprocessing:
    def __init__(self, data=None):
        nltk.download('averaged_perceptron_tagger')
        self.tagger = PerceptronTagger()
        self.lemmatizer = WordNetLemmatizer()
        self.data = data

    def run_cleaner(self):
        for column in self.data.columns:
            self.data[column] = self.data[column].apply(word_tokenize)
            #self.data[column] = self.data[column].apply(self.auto_correct)

    def run_meaning(self):
        for column in self.data.columns:
            pass
            self.data[column] = self.data[column].apply(self.remove_stop_words)
            #self.data[column] = self.data[column].apply(self.tagger.tag)
            #self.data[column] = self.data[column].apply(self.lemmatize)
            #self.data[column] = self.data[column].apply(self.meaning)
            #self.data[column] = self.data[column].apply(self.revectorize)
            # Join toghether names ?
            # Remove stopwords ?

    def as_string(self):
        string_df = pd.DataFrame(columns=['sentence0', 'sentence1'])
        for column in self.data.columns:
            string_df[column] = self.data[column].str.join(' ')
        return string_df

    def save_dump(self, name):
        self.data.to_pickle(name)

    def load_dump(self, name):
        self.data = pd.read_pickle(name)

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

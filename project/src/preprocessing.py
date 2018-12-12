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
        self.run_lemmas(copy)
        self.run_meaning(copy)
        # Remove all final 's' in words
        self.as_string(copy)
        return copy

    def run_cleaner(self, data):
        for column in data.columns:
            data[column] = data[column].apply(word_tokenize)
            # data[column] = data[column].apply(self.auto_correct)

    def run_lemmas(self, data):
        for column in data.columns:
            data[column] = data[column].apply(self.remover) # Removes stop words and symbols
            data[column] = data[column].apply(self.tagger.tag)
            data[column] = data[column].apply(self.lemmatize)
            # Join together names ?

    def run_meaning(self, data):
        # Change numbers to text
        for column in data.columns:
            pass
            # data[column] = data[column].apply(self.dt_noun_joiner)
        #for index, row in data.iterrows():
        #    data[column] = data[column].apply(self.meaning)
        for column in data.columns:
            data[column] = data[column].apply(self.revectorize)

    def as_string(self, data, to_lower=True):
        for column in data.columns:
            if to_lower:
                data[column] = data[column].str.join(' ')
                data[column] = data[column].str.lower()
            else:
                data[column] = data[column].str.join(' ')

    def remover(self, vector):
        new_vector = []
        for word in vector:
            word = re.sub(r'\.\d+', '', word)  # Remove decimals
            word = re.sub(r'\W+', '', word)  # Remove symbols
            word = re.sub(r'\s+', ' ', word)  # Replace multiple spaces by one
            #print(word)
            #if not word in ['\'s'] and not word in list(stopwords.words('english')):
            if len(word) > 0 and not word in list(stopwords.words('english')):
                #print(word)
                new_vector.append(word)
        return new_vector

    def revectorize(self, tagged):
        return [word for word, tag in tagged]

    def auto_correct(self, vector):
        return [spell(word) for word in vector]

    def penn_to_wn(self, tag):
        if tag in ['JJ', 'JJR', 'JJS']:
            return wn.ADJ
        elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return wn.NOUN
        elif tag in ['RB', 'RBR', 'RBS']:
            return wn.ADV
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wn.VERB
        return wn.NOUN

    def lemmatize(self, tagged):
        result = []
        for word, tag in tagged:
            if word.endswith('ing'): # A verb for sure (?). Sometimes, verbs are wrongly classified, for example, "A man is smoking" (Smoking => Noun)
                tag = 'VB'
            lemma = self.lemmatizer.lemmatize(word, self.penn_to_wn(tag))
            result.append((lemma, tag))
        return result

    def dt_noun_joiner(self, tagged):
        result = []
        for i in range(len(tagged) - 1):
            if tagged[i][1] != 'DT' or not tagged[i + 1][1].startswith('N'):
                result.append(tagged[i])
        result.append(tagged[-1])
        return result


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

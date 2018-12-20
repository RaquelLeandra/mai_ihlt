from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import nltk
import re
import os.path
import pandas as pd

from nltk.corpus import stopwords
from lex_path import lex_compare


class Preprocessor:
    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        self.tagger = PerceptronTagger()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = list(stopwords.words('english'))
        self.auto_correct_remaining = 0

    def run(self, data):
        data = data.copy()

        self.auto_correct_remaining = len(data.index) * len(data.columns)
        for column in data.columns:
            data[column] = data[column].apply(word_tokenize)
            #data[column] = data[column].apply(self.auto_correct)
            data[column] = data[column].apply(self.remover)  # Remove stop words and symbols
            data[column] = data[column].apply(self.tagger.tag)
            data[column] = data[column].apply(self.lemmatize)
        print()

        for _, row in data.iterrows():
            self.meaning(row)

        for column in data.columns:
            data[column] = data[column].apply(self.revectorize)

        return data

    def revectorize(self, tagged): # And remove final s
        return [re.sub('s$', '', word).lower() for word, tag in tagged]

    # ------------------------------------------------ REMOVE & FIX ▼ --------------------------------------------------

    def remover(self, vector):
        new_vector = []
        for word in vector:
            word = re.sub(r'\.\d+', '', word)  # Remove decimals
            word = re.sub(r'\W+', '', word)  # Remove symbols
            word = re.sub(r'\s+', ' ', word)  # Replace multiple spaces by one
            word = re.sub(r'^\s+|\s+$', '', word) # Trim spaces

            if word and not (word in self.stopwords): # TODO word.lower() in self.stopwords
                new_vector.append(word)

        return new_vector

    def auto_correct(self, vector):
        self.auto_correct_remaining -= 1
        print('\rSpell auto-correct...', self.auto_correct_remaining, 'sentences remain', end='    ', flush=True)
        return [spell(word) for word in vector]

    # --------------------------------------------------- LEMMAS ▼ ----------------------------------------------------

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
            # A verb for sure if it ends in ing (?).
            # Sometimes, verbs are wrongly classified, for example, "A man is smoking" (Smoking => Noun)
            if word.endswith('ing'):
                tag = 'VB'
            lemma = self.lemmatizer.lemmatize(word, self.penn_to_wn(tag))
            result.append((lemma, tag))
        return result

    # --------------------------------------------------- MEANING ▼ ----------------------------------------------------

    def common_start(self, A, B):
        count = 0
        for a, b in zip(A, B):
            if a != b:
                return count
            count += 1
        return count

    def common_end(self, A, B):
        count = 0
        for a, b in zip(reversed(A), reversed(B)):
            if a != b:
                return count
            count += 1
        return count

    def get_synsets(self, word, tag):
        synsets = wn.synsets(word, self.penn_to_wn(tag))
        return [str(synset).replace('Synset(\'', '').replace('\')', '') for synset in synsets]

    def meaning(self, row):

        s1 = row['sentence0']
        s2 = row['sentence1']

        for i1, (token1, tag1) in enumerate(s1):
            for i2, (token2, tag2) in enumerate(s2):
                if tag1 != 'DT' and tag2 != 'DT' and token1 != token2 and not self.is_number(token1) and not self.is_number(token2):
                    # Check common starting
                    if  self.common_start(token1, token2) > 2 or self.common_end(token1, token2) > 3:
                        #print('CHANGEA', s1[i1], s2[i2], "to", token1)
                        s1[i1] = (token1, tag1)
                        s2[i2] = (token1, tag2)
                        pass
                    elif tag1.startswith('V') and tag2.startswith('V'):
                        synset1 = wn.synsets(token1, self.penn_to_wn(tag1))
                        synset2 = wn.synsets(token2, self.penn_to_wn(tag2))
                        if len(synset1) > 0 and len(synset2) > 0:
                            full_path, min_common, min_dist = lex_compare(
                                str(synset1[0]).replace('Synset(\'', '').replace('\')', ''),
                                str(synset2[0]).replace('Synset(\'', '').replace('\')', ''))

                            if  min_common is not None \
                                and min_dist < 4 \
                                and not str(min_common).startswith('Synset(\'entity') \
                                and not str(min_common).startswith('Synset(\'abstraction'):
                                    #print('CHANGEB', s1[i1], s2[i2], "to", end=' ')
                                    s1[i1] = (re.sub(r'Synset\(\'(.*)\..*\..*', r'\1', str(min_common)), tag1)
                                    s2[i2] = (re.sub(r'Synset\(\'(.*)\..*\..*', r'\1', str(min_common)), tag2)
                                    #print(s1[i1], min_dist)
        return row

    def is_number(self, s):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

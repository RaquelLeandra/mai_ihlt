from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from nltk.tag import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import nltk
import re

from nltk.corpus import stopwords
from lex_path import lex_compare


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
        for index, row in data.iterrows():
            row = self.meaning(row)
        for column in data.columns:
            data[column] = data[column].apply(self.revectorize)

    def as_string(self, data, to_lower=True):
        for column in data.columns:
            if to_lower:
                data[column] = data[column].str.join(' ')
                data[column] = data[column].str.lower()
            else:
                data[column] = data[column].str.join(' ')

            # Remove common articles
            data[column] = data[column].apply(self.remove_common)

    def remove_common(self, sentence):
        #return re.sub(r'(the|a|one|each|only)\s+(girl|boy|man|woman|men|women|person|adult|people)', 'a \\1', sentence)
        return sentence

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

    def meaning(self, row):

        s1 = row['sentence0']
        s2 = row['sentence1']

        for i1, (token1, tag1) in enumerate(s1):
            for i2, (token2, tag2) in enumerate(s2):
                if tag1 != 'DT' and tag2 != 'DT' and token1 != token2 and not self.is_number(token1) and not self.is_number(token2):
                    # Check common starting
                    if  self.common_start(token1, token2) > 2 or self.common_end(token1, token2) > 2:
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
                                    print('CHANGEB', s1[i1], s2[i2], "to", end=' ')
                                    s1[i1] = (re.sub(r'Synset\(\'(.*)\..*\..*', r'\1', str(min_common)), tag1)
                                    s2[i2] = (re.sub(r'Synset\(\'(.*)\..*\..*', r'\1', str(min_common)), tag2)
                                    print(s1[i1], min_dist)
        return row

    def is_number(self, s):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True
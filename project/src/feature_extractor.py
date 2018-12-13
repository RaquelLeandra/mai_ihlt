import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.metrics import jaccard_distance
from nltk.tag import PerceptronTagger
from nltk import pos_tag
import string

class FeatureExtractor:
    def __init__(self):
        pass

    def extract(self, dataset):

        features = pd.DataFrame(columns=['sentence_0_lengh','sentence_1_lengh',
                                        'number_of_nouns_s0', 'number_of_nouns_s1',
                                        'number_of_verbs_s0', 'number_of_verbs_s1',
                                        'number_of_symbols_s0','number_of_symbols_s1',
                                       'number_of_digits_s0','number_of_digits_1',
                                       'quantity_of_synonims','quantity_of_shared_words',
                                        'proper_nouns_shared','jaccard_distance','path_similarity',
                                        'wup_similarity'])

        mx = len(dataset.index)
        for index, row in dataset.iterrows():
            s0 = row['sentence0']
            s1 = row['sentence1']
            print('\rExtracting features...', index, 'of', mx, end='   ', flush=True)
            features.loc[index,'jaccard_distance'] = self.calculate_jaccard(s0,s1)
            features.loc[index,'path_similarity'] = self.sentence_similarity(s0,s1,wn.path_similarity)
            features.loc[index,'wup_similarity'] = self.sentence_similarity(s0,s1,wn.wup_similarity)
            features.loc[index,'proper_nouns_shared'] = self.count_common_propper_nouns(s0,s1)
            features.loc[index,'quantity_of_shared_words'] = self.count_shared_words(s0,s1)
            features.loc[index,'quantity_of_synonims'] = self.count_synonims(s0,s1)
            features.loc[index,'sentence_0_lengh'] = self.sentence_lenght(s0)
            features.loc[index,'sentence_1_lengh'] = self.sentence_lenght(s1)
            features.loc[index,'number_of_nouns_s0'] = self.count_nouns(s0)
            features.loc[index,'number_of_nouns_s1'] = self.count_nouns(s1)
            features.loc[index,'number_of_verbs_s0'] = self.count_verbs(s0)
            features.loc[index,'number_of_verbs_s1'] = self.count_verbs(s1)
            features.loc[index,'number_of_symbols_s0'] = self.count_symbols(s0)
            features.loc[index,'number_of_symbols_s1'] = self.count_symbols(s1)
            features.loc[index,'number_of_digits_s0'] = self.count_digits(s0)
            features.loc[index,'number_of_digits_1'] = self.count_digits(s1)
        print()

        return features

    def sentence_lenght(self, s):
        return len(s)

    def count_symbols(self, s):
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        return count(s, set(string.punctuation))

    def count_shared_words(self, s0, s1):
        s0 = [w.lower() for w in s0]
        s1 = [w.lower() for w in s1]
        return len(list(set(s0) & set(s1)))

    def count_digits(self, s):
        numbers = sum(c.isdigit() for c in s)
        return numbers

    def _get_word_synonyms(self, word):
        word_synonyms = []
        for synset in wn.synsets(word):
            for lemma in synset.lemma_names():
                word_synonyms.append(lemma)
        return word_synonyms

    def synonim_words(self, a, b):
        return len(set(self._get_word_synonyms(a)) & set(self._get_word_synonyms(b))) > 0

    def count_synonims(self, s0, s1):
        sinonim = 0
        for a in s0:
            for b in s1:
                sinonim += self.synonim_words(a.lower(), b.lower())
        return sinonim

    def count_common_propper_nouns(self, s0, s1):
        tagger = PerceptronTagger()
        s0_tags = tagger.tag(s0)
        s1_tags = tagger.tag(s1)
        NNP_s0 = [values[0] for values in s0_tags if values[1] == 'NNP']
        NNP_s1 = [values[0] for values in s1_tags if values[1] == 'NNP']
        return len(set(NNP_s0) & set(NNP_s1))

    def count_nouns(self, s0):
        tagger = PerceptronTagger()
        s0_tags = tagger.tag(s0)
        NN_s0 = [values[0] for values in s0_tags if values[1] == 'NN']
        return len(NN_s0)

    def count_verbs(self, s0):
        tagger = PerceptronTagger()
        s0_tags = tagger.tag(s0)
        V_s0 = [values[0] for values in s0_tags if values[1] == 'VBP']
        return len(V_s0)

    def remove_stop_words(self, s0):
        new_vector = ' '.join(word for word in s0.split() if word.lower() not in list(stopwords.words('english')))
        return new_vector

    def calculate_jaccard(self, s0, s1):
        lemms_0 = set([a.lower() for a in s0 if a])
        lemms_1 = set([a.lower() for a in s1 if a])

        jaccard_simmilarity = (1 - jaccard_distance(lemms_0, lemms_1))
        return jaccard_simmilarity

    def penn_to_wn(self, tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'

        if tag.startswith('V'):
            return 'v'

        if tag.startswith('J'):
            return 'a'

        if tag.startswith('R'):
            return 'r'
        return 'n'

    def tagged_to_synset(self, word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None
        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None

    def sentence_similarity(self, sentence1, sentence2, similarity=wn.path_similarity):
        """ compute the sentence similarity using Wordnet """
        # Tokenize and tag
        sentence1 = pos_tag(sentence1)
        sentence2 = pos_tag(sentence2)

        # Get the synsets for the tagged words
        synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence2]

        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]

        score, count = 0.0, 0

        # For each word in the first sentence
        for synset in synsets1:
            # Get the similarity value of the most similar word in the other sentence
            similarities = [similarity(synset, ss) for ss in synsets2 if similarity(synset, ss)]
            try:
                best_score = max(similarities)
            except:
                best_score = 0
            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1

        # Average the values
        try:
            score /= count
        except:
            score = 0
        return score

from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
from nltk import pos_tag
import nltk


class Jaccard:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __lemmatize(self, sentence):
        words = nltk.word_tokenize(sentence)
        pairs = pos_tag(words)
        lemms = []
        for pair in pairs:
            if pair[1][0] in {'N', 'V'}:
                lemms.append(self.wnl.lemmatize(pair[0].lower(), pos=pair[1][0].lower()))
            else:
                lemms.append(pair[0].lower())
        return lemms

    def predict(self, data_frame, maximum=5):
        predicted = []
        for index, row in data_frame.iterrows():
            s1 = row['sentence0']
            s2 = row['sentence1']
            lemms1 = set(self.__lemmatize(s1))
            lemms2 = set(self.__lemmatize(s2))
            jaccard_similarity = (1 - jaccard_distance(lemms1, lemms2)) * maximum
            predicted.append(jaccard_similarity)

        return predicted
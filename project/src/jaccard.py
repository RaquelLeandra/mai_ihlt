from nltk.metrics import jaccard_distance
import nltk


class Jaccard:
    def __init__(self):
        pass

    def predict(self, data_frame, maximum=5):
        predicted = []
        for index, row in data_frame.iterrows():
            s1 = row['sentence0']
            s2 = row['sentence1']
            lemms1 = set(nltk.word_tokenize(s1))
            lemms2 = set(nltk.word_tokenize(s2))
            jaccard_similarity = (1 - jaccard_distance(lemms1, lemms2)) * maximum
            predicted.append(jaccard_similarity)

        return predicted
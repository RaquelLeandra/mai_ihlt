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
            jaccard_similarity = (1 - jaccard_distance(set(s1), set(s2))) * maximum
            predicted.append(jaccard_similarity)

        return predicted
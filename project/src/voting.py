from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
from nltk import pos_tag
import nltk
import numpy as np

wnl = WordNetLemmatizer()

def lemmatize(sentence):
    words = nltk.word_tokenize(sentence)
    pairs = pos_tag(words)
    lemms = []
    for pair in pairs:
        if pair[1][0] in {'N', 'V'}:
            lemms.append(wnl.lemmatize(pair[0].lower(), pos=pair[1][0].lower()))
        else:
            lemms.append(pair[0].lower())
    return lemms



def fix(test_predicted, test):
    for index, row in test.iterrows():
        s1 = row['sentence0']
        s2 = row['sentence1']
        lemms_1 = set(lemmatize(s1))
        lemms_2 = set(lemmatize(s2))
        jaccard_simmilarity = (1 - jaccard_distance(lemms_1, lemms_2)) * 5
        if abs(test_predicted[index] - jaccard_simmilarity) > 2:
            test_predicted[index] = np.mean([jaccard_simmilarity, test_predicted[index]])

    return test_predicted
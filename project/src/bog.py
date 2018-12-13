import gensim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class BOG:

    def __init__(self):
        self.__dictionary = None
        self.scaler = StandardScaler()

    def train_dictionary(self, tokenized_trn):
        self.__dictionary = gensim.corpora.Dictionary(tokenized_trn.sentence0.tolist() + tokenized_trn.sentence1.tolist())
        self.__dictionary.filter_extremes(no_below=5, no_above=0.8)
        self.__dictionary.compactify()
        print("BOG dictionary size: %s" % len(self.__dictionary))

    def __get_vectors(self, df):
        sentence0_vec = [self.__dictionary.doc2bow(text) for text in df.sentence0.tolist()]
        sentence1_vec = [self.__dictionary.doc2bow(text) for text in df.sentence1.tolist()]
        sentence0_csc = gensim.matutils.corpus2csc(sentence0_vec, num_terms=len(self.__dictionary.token2id))
        sentence1_csc = gensim.matutils.corpus2csc(sentence1_vec, num_terms=len(self.__dictionary.token2id))
        return sentence0_csc.transpose(), sentence1_csc.transpose()

    def get_bog_extended(self, tokenized, features, scale=False):
        q1_csc, q2_csc = self.__get_vectors(tokenized)
        trn_bog = np.concatenate((q1_csc.todense(), q2_csc.todense()), axis=1)
        if scale:
            features = pd.DataFrame(self.scaler.fit_transform(features))
        trn_bog_extended = pd.concat([pd.DataFrame(trn_bog), features], axis=1)
        return trn_bog_extended

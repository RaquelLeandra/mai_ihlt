from sklearn.ensemble import RandomForestRegressor
import gensim
import pandas as pd
import numpy as np


class RFR(RandomForestRegressor):

    def __init__(self, n_jobs=-1, n_estimators=100, bag_of_words=True):
        super().__init__(n_jobs=n_jobs, n_estimators=n_estimators)
        self.bag_of_words = bag_of_words

    def __train_dictionary(self, df):
        self.dictionary = gensim.corpora.Dictionary(df.sentence0.tolist() + df.sentence1.tolist())
        self.dictionary.filter_extremes(no_below=5, no_above=0.8)
        self.dictionary.compactify()
        print("BOG dictionary size: %s" % len(self.dictionary))

    def __get_vectors(self, df):
        sentence0_vec = [self.dictionary.doc2bow(text) for text in df.sentence0.tolist()]
        sentence1_vec = [self.dictionary.doc2bow(text) for text in df.sentence1.tolist()]
        sentence0_csc = gensim.matutils.corpus2csc(sentence0_vec, num_terms=len(self.dictionary.token2id))
        sentence1_csc = gensim.matutils.corpus2csc(sentence1_vec, num_terms=len(self.dictionary.token2id))
        return sentence0_csc.transpose(), sentence1_csc.transpose()

    def train_bog(self, tokenized, features):
        q1_csc, q2_csc = self.__get_vectors(tokenized)
        trn_bog = np.concatenate((q1_csc.todense(), q2_csc.todense()), axis=1)
        trn_bog_extended = pd.concat([pd.DataFrame(trn_bog), features], axis=1)
        return trn_bog_extended

    def train(self, tokenized_trn, features_trn, labels):
        self.__train_dictionary(tokenized_trn)
        trn_bog_extended = self.train_bog(tokenized_trn, features_trn)
        self.fit(trn_bog_extended, labels)
        self.print_feature_importance(trn_bog_extended)

    def predict_(self, tokenized, features):
        return self.predict(self.train_bog(tokenized, features))

    def print_feature_importance(self, trn):
        importance = self.feature_importances_
        indices = np.argsort(importance)[::-1]
        try:
            feat_labels = trn.columns
            for f in range(10):
                print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))
        except:
            pass
import pandas as pd
import numpy as np
import csv
from os import listdir, path as pth
from beautifultable import BeautifulTable

from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

from preprocessing import Preprocessor
from jaccard import Jaccard

class Classifier:
    _GS_COLS = ['labels']
    _COLS = ['sentence0', 'sentence1']
    _DUMP_FILES = {
        'TRN': './dump/classifier.trn.dump',
        'TST': './dump/classifier.tst.dump',
        'TRN_GS': './dump/classifier.trn_gs.dump',
        'TST_GS': './dump/classifier.tst_gs.dump'
    }

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor = Preprocessor()
        self.trn = pd.DataFrame(columns=Classifier._COLS)       # Read data_frame
        self.tst = pd.DataFrame(columns=Classifier._COLS)       # Read data_frame
        self.trn_gs = pd.DataFrame(columns=Classifier._GS_COLS) # Known labels
        self.tst_gs = pd.DataFrame(columns=Classifier._GS_COLS) # Known labels
        self.pre_trn = pd.DataFrame(columns=Classifier._COLS)   # Preprocessed data_frame
        self.pre_tst = pd.DataFrame(columns=Classifier._COLS)   # Preprocessed data_frame
        self.vec_trn = []
        self.vec_tst = []

        self.jaccard = Jaccard()
        self.rfr = RandomForestRegressor(n_jobs=-1)
        self.vectorizer = TfidfVectorizer(max_features=None,
                                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                          stop_words='english')

    # -------------------------------------------------- CLASSIFY ▼ ----------------------------------------------------

    def classify(self):
        print(self.trn.head())
        print('Preprocessing...')
        self.pre_trn = self.preprocessor.run(self.trn)
        self.pre_tst = self.preprocessor.run(self.tst)
        self.vec_trn = self.vectorizer.fit_transform(self.pre_trn['sentence0'] + self.pre_trn['sentence1'])
        self.vec_tst = self.vectorizer.transform(self.pre_tst['sentence0'] + self.pre_tst['sentence1'])
        print(self.pre_trn.head())

        print('Training...')
        self.train_rfr()

        print('Testing...')
        predict_rfr_trn, predict_rfr_tst = self.test_rfr()
        predict_jac_trn, predict_jac_tst = self.test_jac()
        predict_trn = self.voting(predict_rfr_trn, predict_jac_trn)
        predict_tst = self.voting(predict_rfr_tst, predict_jac_tst)

        print('Done !')
        self.show_results(predict_rfr_trn, predict_rfr_tst, predict_jac_trn, predict_jac_tst, predict_trn, predict_tst)

    def voting(self, predict_rfr, predict_jac):
        voted = []
        for rfr, jac in zip(predict_rfr, predict_jac):
            if abs(rfr - jac) > 2:
                voted.append(np.mean([jac, rfr]))
            else:
                voted.append(rfr)
        return voted

    # --------------------------------------------------- MODELS ▼ ----------------------------------------------------

    # Random Forest Regression
    def train_rfr(self):
        self.rfr.fit(self.vec_trn, self.trn_gs.values.ravel())

    def test_rfr(self):
        return self.rfr.predict(self.vec_trn), self.rfr.predict(self.vec_tst)

    def test_jac(self):
        return self.jaccard.predict(self.pre_trn), self.jaccard.predict(self.pre_tst)

    # ---------------------------------------------------- SHOW ▼ -----------------------------------------------------

    def __add_table(self, table, name, trn, tst):
        table.append_column(name, [
            '{:.2f} std: {:.1f}'.format(np.mean(trn), np.std(trn)),
            '{:.2f} std: {:.1f}'.format(np.mean(tst), np.std(tst)),
            '{:.4f}'.format(pearsonr(trn, self.trn_gs['labels'])[0]),
            '{:.4f}'.format(pearsonr(tst, self.tst_gs['labels'])[0])
        ])

    def show_results(self, rfr_trn, rfr_tst, jac_trn, jac_tst, trn, tst):
        table = BeautifulTable()
        table.append_column('', ['Trn', 'Tst', 'Trn Pearson', 'Tst Pearson'])

        self.__add_table(table, 'Real', self.trn_gs['labels'], self.tst_gs['labels'])
        self.__add_table(table, 'RFR', rfr_trn, rfr_tst)
        self.__add_table(table, 'Jaccard', jac_trn, jac_tst)
        self.__add_table(table, 'Voting', trn, tst)
        print(table)
        self.show_worst_test(tst)

    def show_worst_test(self, predicted, k=5):
        print('Worst results in voting:')
        err = np.abs(predicted - self.tst_gs['labels'].values)
        idx = np.argpartition(err, -k)[-k:]
        for i in idx:
            print(
                'Predicted: {:.2f} Target: {:.2f} Err: {:.2f}\nOriginal: [{:s}] [{:s}] Preprocessed:\n[{:s}] [{:s}]'
                .format(
                    predicted[i], self.tst_gs['labels'].values[i], err[i],
                    str(self.tst['sentence0'].values[i]).replace('\n', '').replace('\r', ''),
                    str(self.tst['sentence1'].values[i]).replace('\n', '').replace('\r', ''),
                    str(self.pre_tst['sentence0'].values[i]),
                    str(self.pre_tst['sentence1'].values[i]),
                ))

    # --------------------------------------------------- LOADING ▼ ---------------------------------------------------

    def load(self, use_dump=False):
        #  Load from txt files
        if not use_dump:
            self.trn, self.trn_gs = self.__load_all(self.train_path)
            self.tst, self.tst_gs = self.__load_all(self.test_path)
        #  Load from dump
        else:
            self.trn = pd.read_pickle(Classifier._DUMP_FILES['TRN'])
            self.tst = pd.read_pickle(Classifier._DUMP_FILES['TST'])
            self.trn_gs = pd.read_pickle(Classifier._DUMP_FILES['TRN_GS'])
            self.tst_gs = pd.read_pickle(Classifier._DUMP_FILES['TST_GS'])

        print('Train: {0} Test: {1}'.format(self.trn.shape, self.tst.shape))

    def save_dump(self):
        self.trn.to_pickle(Classifier._DUMP_FILES['TRN'])
        self.tst.to_pickle(Classifier._DUMP_FILES['TST'])
        self.trn_gs.to_pickle(Classifier._DUMP_FILES['TRN_GS'])
        self.tst_gs.to_pickle(Classifier._DUMP_FILES['TST_GS'])

    def __load_all(self, dir):
        files = listdir(dir)
        input = pd.DataFrame(columns=['sentence0', 'sentence1'])
        label = pd.DataFrame(columns=['labels'])
        for file in files:
            path = pth.join(dir, file)
            path_gs = path.replace('input', 'gs')
            if 'STS.input' in path:  # Only read input files
                input_df = pd.read_csv(path, sep='\t', lineterminator='\n', names=Classifier._COLS, header=None, quoting=csv.QUOTE_NONE)
                label_df = pd.read_csv(path_gs, sep='\t', lineterminator='\n', names=Classifier._GS_COLS, header=None, quoting=csv.QUOTE_NONE)
                input = input.append(input_df)
                label = label.append(label_df)

        return \
            input.fillna('').reset_index(drop=True), \
            label.fillna('').reset_index(drop=True)
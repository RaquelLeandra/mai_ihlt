from load import load_all, load_gs
from preprocessing import Preprocessing

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


TRAIN_PATH = '../data/train/'
TEST_PATH = '../data/test-gold/'

if __name__ == '__main__':
    #train, train_gs = load_all(TRAIN_PATH)
    #test, test_gs = load_all(TEST_PATH)
    #print('Train: {0} Test: {1}'.format(train.shape, test.shape))
    train_gs = load_gs(TRAIN_PATH)
    test_gs = load_gs(TEST_PATH)

    def test_model(model,xtrain,xtest):
        train_predicted =  model.predict(xtrain)
        test_predicted =   model.predict(xtest)
        print('Train real', 'Avg:', np.mean(train_gs['labels']), 'Std:', np.std(train_gs['labels']))
        print('Train predicted', 'Avg:', np.mean(train_predicted), 'Std:', np.std(train_predicted))
        print('Test real', 'Avg:', np.mean(test_gs['labels']), 'Std:', np.std(test_gs['labels']))
        print('Test predicted', 'Avg:', np.mean(test_predicted), 'Std:', np.std(test_predicted))
        print('train pearson: ', pearsonr(train_predicted, train_gs['labels'])[0])
        print('test pearson: ', pearsonr(test_predicted, test_gs['labels'])[0])

    def train_and_test_model(model, train,test):
        model.fit(train,train_gs)
        test_model(model,train,test)

    preprocessing_train = Preprocessing()
    preprocessing_test = Preprocessing()
    preprocessing_train.load_dump('./train.dump')
    preprocessing_test.load_dump('./test.dump')
    train = preprocessing_train.data
    test = preprocessing_test.data
    print('Train: {0} Test: {1}'.format(train.shape, test.shape))
    print(train.head())
    print('Processing...')
    #preprocessing_train = Preprocessing(train)
    #preprocessing_test = Preprocessing(test)
    #preprocessing_train.run_cleaner()
    #preprocessing_test.run_cleaner()
    #preprocessing_train.save_dump('./train.dump')
    #preprocessing_test.save_dump('./test.dump')
    preprocessing_train.run_meaning()
    preprocessing_test.run_meaning()
    train = preprocessing_train.as_string()
    test = preprocessing_test.as_string()
    print(train.head())

    merged_train = train['sentence0'] + train['sentence1']
    merged_test = test['sentence0'] + test['sentence1']

    vectorizer = TfidfVectorizer(max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                 ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words='english')

    merged_train = vectorizer.fit_transform(merged_train)
    merged_test = vectorizer.transform(merged_test)

    print(merged_train.shape)
    print(merged_test.shape)
    rfr = RandomForestRegressor(n_jobs=-1)
    train_and_test_model(rfr, merged_train, merged_test)

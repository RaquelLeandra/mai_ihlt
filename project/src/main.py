from load import load_all
from preprocessing import preprocess

from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


TRAIN_PATH = '../data/train/'
TEST_PATH = '../data/test-gold/'

if __name__ == '__main__':
    train, train_gs = load_all(TRAIN_PATH)
    test, test_gs = load_all(TEST_PATH)
    print('Train: {0} Test: {1}'.format(train.shape, test.shape))

    def test_model(model,xtrain,xtest):
        train_predicted =  model.predict(xtrain)
        test_predicted =   model.predict(xtest)
        print('train pearson: ', pearsonr(train_predicted, train_gs['labels'])[0])
        print('test pearson: ', pearsonr(test_predicted, test_gs['labels'])[0])

    def train_and_test_model(model, train,test):
        model.fit(train,train_gs)
        test_model(model,train,test)

    print(train.head())
    print('Processing...')
    #train = preprocess(train)
    #test = preprocess(test)
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
    rfr = RandomForestRegressor(n_jobs=-1, n_estimators=500)
    train_and_test_model(rfr, merged_train, merged_test)

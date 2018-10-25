from scipy.stats import pearsonr
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.metrics import jaccard_distance
import csv

def load_data():
    train_path = 'data/train/STS.input.MSRpar.txt'
    train_gs_path = 'data/train/STS.gs.MSRpar.txt'
    test_path = 'data/test-gold/STS.input.MSRpar.txt'
    test_gs_path = 'data/test-gold/STS.gs.MSRpar.txt'
    train_df = pd.read_csv(train_path, sep='\t', lineterminator='\n', names=['sentence0','sentence1'], header=None, quoting=csv.QUOTE_NONE)
    train_gs = pd.read_csv(train_gs_path, sep='\t', lineterminator='\n', names=['labels'], header=None)
    print('train shape:', train_df.shape, train_gs.shape)
    return train_df,test_df

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = WhitespaceTokenizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def preprocessing(data):
    # todo: better handling of na
    data = data.fillna('')
    for column in data.columns:
        print(column)
        # remove the digits and puntuation
        data[column] = data[column].str.replace('\d+', '')
        # convert to lowercase
        data[column] = data[column].str.replace('\W+', ' ')
        # replace continuous white spaces by a single one
        data[column] = data[column].str.replace('\s+', ' ')
        # words to lower
        data[column] =data[column].str.lower()
        # lematize
        data[column] = data[column].apply(lemmatize_text)
    return data
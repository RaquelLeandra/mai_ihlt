from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from autocorrect import spell
import re


def lemmatize_text(text):
    """ Convert the text into lemmas """
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = WhitespaceTokenizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def auto_spell(text):
    """ Correct spelling errors """
    return ' '.join(spell(word) for word in text.split())


def punctuation(text):
    """ Remove or change punctuation """
    text = re.sub('\d+', '', text)
    text = re.sub('\W+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text


def preprocess(data, as_vector=True):
    data = data.fillna('')
    for column in data.columns:
        # remove the digits and punctuation
        data[column] = data[column].apply(punctuation)
        # words to lower
        data[column] = data[column].str.lower()
        # spell corrector
        data[column] = data[column].apply(auto_spell)
        # lemmatize
        data[column] = data[column].apply(lemmatize_text)
        # convert vector to string
        if not as_vector:
            data[column] = data[column].str.join(' ')
    return data

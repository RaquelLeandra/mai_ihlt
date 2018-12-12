from classifier import Classifier

if __name__ == '__main__':
    classifier = Classifier(train_path='../data/train/', test_path='../data/test-gold/')
    classifier.load()
    classifier.classify()

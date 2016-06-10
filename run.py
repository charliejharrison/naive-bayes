__author__ = 'charlie'

import pickle

from preprocess import *

from time import time

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from sklearn.cross_validation import LabelKFold


def print_result(_result, _i):
    """
    Modular printing for easier debugging.  Add whatever output you want to see
    """
    print "##########"
    print "Fold {}\taccuracy = {}".format(_i, _result['accuracy'])
    _result['classifier'].show_most_informative_features(10)
    print "##########"


if __name__ == '__main__':
    # TODO: smoothing?  Don't think this is done automatically by NLTK

    print "Loading data..."

    num_words = None

    features, labels = load_data_and_labels(num_words)
    folds = LabelKFold(labels, 10)

    results = []
    i = 0

    print "Training models..."
    for train_inds, test_inds in folds:
        i += 1
        result = {}
        train_set = features[train_inds]
        test_set = features[test_inds]

        classifier = NaiveBayesClassifier.train(train_set)

        result['accuracy'] = accuracy(classifier, test_set)
        result['train_inds'] = train_inds
        result['test_inds'] = test_inds
        result['classifier'] = classifier

        results.append(result)
        print_result(result, i)

    timestamp = int(time() * 1000)
    results_fi = 'naive_bayes_results-{}.pickle'.format(timestamp)
    results_file = os.path.join('results', results_fi)

    print "Saving results to {}".format(results_file)
    with open(results_file, 'w+') as fi:
        pickle.dump(results, fi)

    print "Done!"

from time import time

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk import ConfusionMatrix
from sklearn.cross_validation import StratifiedKFold

import numpy as np
import os.path
from os import makedirs
import pickle

import preprocess


def print_result(_result):
    """
    Modular printing for easier debugging.  Add whatever output you want to see
    """
    #_result['classifier'].show_most_informative_features(10)

    #print("Confusion Matrix:")
    #print(_result['confusion'].pretty_format())

    print("\tAccuracy = {}".format(_result['accuracy']))
    print("\tPrecision = {}".format(_result['precision']))
    print("\tRecall = {}".format(_result['recall']))
    print("\tF-score = {}".format(_result['F-score']))
    print("\tType 1 err = {}".format(_result['Type 1 err']))
    print("\tType 2 err = {}".format(_result['Type 2 err']))

    print("\n##########\n")


def train_classifiers(_labelled_features, _folds):
    """
    Train a set of ten classifiers and save their results in dicts.

    :param _labelled_features:
    :param _folds:
    :return:
    """
    _results = []
    i = 0

    for train_inds, test_inds in _folds:
        i += 1
        result = {}
        train_set = _labelled_features[train_inds]
        test_set = _labelled_features[test_inds]

        classifier = NaiveBayesClassifier.train(train_set)

        result['accuracy'] = accuracy(classifier, test_set)
        result['train_inds'] = train_inds
        result['test_inds'] = test_inds
        result['classifier'] = classifier

        pred = []
        test_labels = list(test_set.transpose()[1])
        for t in test_set:
            pred.append(classifier.classify(t[0]))

        confusion = ConfusionMatrix(test_labels, pred)
        tp = confusion['p', 'p']
        tn = confusion['n', 'n']
        fp = confusion['n', 'p']
        fn = confusion['p', 'n']
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        result['test_labels'] = test_labels
        result['pred_labels'] = pred
        result['confusion'] = confusion
        result['precision'] = precision
        result['recall'] = recall
        result['F-score'] = (2 * precision * recall) / (precision + recall)
        result['Type 1 err'] = fp / (tp + fp)
        result['Type 2 err'] = fn / (tn + fn)

        _results.append(result)

    return _results


def pickle_results(_results):
    """
    Serialise and save a trained classifier result using pickle.

    :param _results:
    :return:
    """

    timestamp = int(time() * 1000)
    results_fi = 'naive_bayes_results-{}.pickle'.format(timestamp)
    results_file = os.path.join('results', results_fi)
    results_dir = os.path.dirname(results_file)
    if not os.path.exists(results_dir):
        makedirs(results_dir)

    print("Saving results to {}".format(results_file))
    with open(results_file, 'wb') as fi:
        pickle.dump(_results, fi)


if __name__ == '__main__':
    # TODO: smoothing?  Don't think this is done automatically by NLTK
    print("Loading data...")

    num_words = None

    features, labels = preprocess.load_data_and_labels(num_words)
    folds = StratifiedKFold(labels, 10)

    labelled_features = zip(features, labels)
    labelled_features = np.array(list(labelled_features))

    results = []
    i = 0

    print("Training models...")
    results = train_classifiers(labelled_features, folds)
    for i, r in enumerate(results):
        print("Fold {}".format(i))
        print_result(r)

    print("Saving results...")
    pickle_results(results)

    print("Done!")

import argparse
import os.path
import warnings

from typing import Optional, List, Tuple

import numpy as np
from nbayes import nbayes, NaiveBayes, evaluate_print_metrics, fetch_metric_data
from logreg import LogReg, logreg
from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType

import util

def combination(data_path: str, lambd: float, bin_number: int, m: float, use_cross_validation: bool = False):
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]
    root_dir = os.sep.join(path[:-1])
    print(file_base, root_dir)
    schema, X, y = parse_c45(file_base, root_dir)
    datasets = ((X, y, X, y),)
    NB = NaiveBayes(schema, m=m)
    nb_predictions = predict_nbayes(NB, schema, datasets, X)
    lr_predictions = predict_logreg(schema, datasets, X)
    y_hat = make_choice(lr_predictions, nb_predictions)
    evaluate_print_metrics(y, y_hat)


def predict_combined(schema, X, y):
    # generates predictions using confidence-weighted predictions from a naive bayes classifier and logistic regression classifier
    nbayes_predictions = predict_nbayes(schema, X, y)
    logreg_predictions = predict_logreg(schema, X, y)


def predict_logreg(schema, X, y, train_size, lambd=0.1):
    # preprocess data for logistic regression classifier
    for i in range(len(schema)):
        feature = schema[i]
        if schema[i].ftype == FeatureType.NOMINAL or schema[i].ftype == FeatureType.BINARY:
            encoded_values = util.encode_values(feature, X[:, i])
            X[:, i] = encoded_values
            schema[i] = Feature(schema[i].name, FeatureType.CONTINUOUS)

    # train a logistic regression classifier on the dataset
    X_train, y_train = X[:train_size], y[:train_size]
    classifier = LogReg(schema, lambd=lambd)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X)
    return predictions


def predict_nbayes(schema, X, y, train_size, bin_count=10):
    # preprocess data for naive bayes classifier
    for i in range(len(schema)):
        feature = schema[i]
        if schema[i].ftype == FeatureType.CONTINUOUS:
            d_feature, d_values = util.split_bins(feature, X[:, i], bin_count)
            schema[i] = d_feature
            X[:, i] = d_values

    # train a naive bayes classifier on the dataset
    X_train, y_train = X[:train_size], y[:train_size]
    classifier = NaiveBayes(schema)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X)

    return predictions


def evaluate_print_metrics(y, y_hat):
    print("Accuracy ", util.accuracy(y, y_hat[:, 0]))
    print("Precision ", util.precision(y, y_hat[:, 0]))
    print("Recall ", util.recall(y, y_hat[:, 0]))
    sorted_confidences = util.sort_confidences(y, y_hat[:, 1])
    print("AUC " , util.area_under_roc(sorted_confidences[:, 0], sorted_confidences[:, 1]))

def lr_preprocess(schema, datasets, X):
    for X_train, y_train, X_test, y_test in datasets:
        for i in range(len(schema)):
            feature = schema[i]
            if schema[i].ftype == FeatureType.NOMINAL or schema[i].ftype == FeatureType.BINARY:
                encoded_values = util.encode_values(feature, X[:, i])
                X[:, i] = encoded_values
                schema[i] = Feature(schema[i].name, FeatureType.CONTINUOUS)

        LR = LogReg(schema, lambd=lambd)
        LR.fit(X_train, y_train)
        lr_predictions = LR.predict(X)
    return lr_predictions


def nb_data(schema, X, y):
    processed_schema = []
    processed_X = np.zeros_like(X)
    for i in range(len(schema)):
        feature = schema[i]
        if schema[i].ftype == FeatureType.CONTINUOUS:
            d_feature, d_values = util.split_bins(feature, X[:, i], bin_number)
            processed_schema[i] = d_feature
            processed_X[:, i] = d_values
        else:
            processed_schema[i] = feature
            processed_X[:, i] = X[:, i]
    return schema, X, y


def nb_preprocess(NB: NaiveBayes, schema, datasets, X):
    for i in range(len(schema)):
        feature = schema[i]
        if schema[i].ftype == FeatureType.CONTINUOUS:
            d_feature, d_values = util.split_bins(feature, X[:, i], bin_number)
            schema[i] = d_feature
            X[:, i] = d_values

    for X_train, y_train, X_test, y_test in datasets:
        NB.fit(X_train, y_train)
        nb_predictions = NB.predict(X)

    return nb_predictions


def make_choice(lr: np.ndarray, nb: np.ndarray):
    y_hat = np.zeros((len(lr), 2))
    for current in range(0, len(lr)):
        if lr[current][1] >= nb[current][1]:
            y_hat[current][0] = lr[current][0]
            y_hat[current][1] = lr[current][1]
        else:
            y_hat[current][0] = nb[current][0]
            y_hat[current][1] = nb[current][1]
    return y_hat


if __name__ == '__main__':
     # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Do logistic regression')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lambd', metavar='lambd', type=float,
                            help='Lambda for constant must be nonnegative')
    parser.add_argument('bin_number', metavar='bin_number', type=int,
                         help='Bin number its sorted into, must be greater than two')
    parser.add_argument('m', metavar='M', type=float,
                         help='M for for the m-estimate')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                         help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True)
    args = parser.parse_args()
    if args.lambd < 0:
        raise argparse.ArgumentTypeError('Lambda must be non negative')
    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv
    lambd = args.lambd
    m = args.m
    bin_number = args.bin_number
    combination(data_path, lambd, bin_number, m, use_cross_validation)

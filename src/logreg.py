import argparse
from dataclasses import dataclass, field
import os.path
from typing import Optional, List, Tuple

import numpy as np
from numba import jit
import scipy.special as scs

from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType

import util

class LogReg(Classifier):
    """Logistic Regression classifier
    """
    def __init__(self, schema: List[Feature], lambd: float):
        self._schema = schema
        self._weights = []
        self._bias = -1
        self._lambd = lambd
        for f in schema:
            if(f.ftype == FeatureType.BINARY or f.ftype == FeatureType.NOMINAL):
                raise ValueError("Nominal and binary features are not supported, please encode before passing")

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Trains the classifier on a set of training data

        Args:
            X (np.ndarray): Feature labels of the training set
            y (np.ndarray): Class labels of the training set
            weights (Optional[np.ndarray], optional): Sample weights. Unimplemented at the moment. Defaults to None.
        """
        # initilize weights and bias to 0
        initial_weights = np.zeros(len(self._schema))
        initial_bias = 0

        # generate MLE parameters on gradient descent on log conditional likelihoods
        self._weights, self._bias = gradient_descent(X, y, initial_weights, initial_bias, self._lambd)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels on a set of test data

        Args:
            X (np.ndarray): Feature labels of the test set

        Returns:
            np.ndarray: Predicted class labels and confidences.
                        y[i, 0] = predicted class label for sample i
                        y[i, 1] = confidence of prediction on sample i
        """
        n_samples = X.shape[0]
        y = np.zeros((n_samples, 2), dtype=np.float32)
        cond_likelihoods = np.dot(X, self._weights) + self._bias
        y = np.array([
            (cond_likelihoods > 0.5).astype(int),
            cond_likelihoods
        ]).T

        return y


@dataclass
class ClassifierMetrics:
    """container class for classifier metrics collected on test data
    """
    accuracy: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    auc_values: List[float] = field(default_factory=list)


def fetch_metric_data(classifier: Classifier, X: np.ndarray, y: np.ndarray, metrics: ClassifierMetrics):
    """Measures metrics (accuracy, precision, recall) and writes it to the ClassifierMetrics container

    Args:
        classifier (LogReg): Classifier to test
        X (np.ndarray): Features of the test set
        y (np.ndarray): Class labels of the test set
        metrics (ClassifierMetrics): metrics container to append to
    """
    y_hat = classifier.predict(X)
    metrics.accuracy.append(util.accuracy(y, y_hat[:, 0]))
    metrics.precision.append(util.precision(y, y_hat[:, 0]))
    metrics.recall.append(util.recall(y, y_hat[:, 0]))
    sorted_confidence = util.sort_confidences(y, y_hat[:, 1])
    metrics.auc_values.append(util.area_under_roc(sorted_confidence[:, 0], sorted_confidence[:, 1]))


def evaluate_print_metrics(metrics: ClassifierMetrics):
    # Calculating average values for metrics
    accuracy_mean = np.mean(metrics.accuracy, dtype=np.float32)
    precision_mean = np.mean(metrics.precision, dtype=np.float32)
    recall_mean = np.mean(metrics.recall, dtype=np.float32)
    auc_mean = np.mean(metrics.auc_values, dtype=np.float32)
    # Calculating standard deviation for metrics
    accuracy_std = np.std(metrics.accuracy, dtype=np.float32)
    precision_std = np.std(metrics.precision, dtype=np.float32)
    recall_std = np.std(metrics.recall, dtype=np.float32)

    # Printing mean & standard deviation values of metrics
    print("Accuracy: ", accuracy_mean, accuracy_std)
    print("Precision: ", precision_mean, precision_std)
    print("Recall: ", recall_mean, recall_std)
    print("AUC: ", auc_mean)


def logreg(data_path: str, lambd: float, use_cross_validation: bool = True):
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    for i in range(len(schema)):
        feature = schema[i]
        if schema[i].ftype == FeatureType.NOMINAL or schema[i].ftype == FeatureType.BINARY:
            encoded_values = util.encode_values(feature, X[:, i])
            X[:, i] = encoded_values
            schema[i] = Feature(schema[i].name, FeatureType.CONTINUOUS)
        # normalize values to [0, 1]
        X[:, i] = X[:, i]/np.max(X[:, i])

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    metrics = ClassifierMetrics()
    for X_train, y_train, X_test, y_test in datasets:
        classifier = LogReg(schema, lambd=lambd)
        classifier.fit(X_train, y_train)
        fetch_metric_data(classifier, X_test, y_test, metrics)
    evaluate_print_metrics(metrics)


def gradient_descent(X, y, weights, bias, lambd, step_size=1):
    # include bias as weight with feature value always 1
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)  # x_0 = 1
    w = np.insert(weights, 0, bias)  # w_0 = bias

    # gradient descent on log conditional likelihood
    gradient = logreg_gradient(X, y, w, lambd)
    while (np.dot(gradient, gradient) > 0.000002):
        # print(f"gradmag: {np.dot(gradient, gradient)}", end="\r")
        w = w - (step_size * gradient)
        step_size *= 0.9999
        gradient = logreg_gradient(X, y, w, lambd)
    bias = w[0]
    weights = w[1:]
    return weights, bias

@jit(forceobj=True)
def logreg_gradient(X, y, w, lambd):
    sigmoidDot = 1/(1+np.exp(-np.dot(X, w)))
    np.dot((sigmoidDot - y), X[:, 0])

    grad_log_likelihoods = np.dot(sigmoidDot-y, X)

    gradient = lambd * w / w.size + grad_log_likelihoods / y.size

    return gradient


if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Do logistic regression')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lambd', metavar='lambd', type=float,
                        help='Lambda for constant must be nonnegative')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')

    parser.set_defaults(cv=True)
    args = parser.parse_args()
    if args.lambd < 0:
        raise argparse.ArgumentTypeError('Lambda must be non negative')
    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv
    lambd = args.lambd
    logreg(data_path, lambd, use_cross_validation)
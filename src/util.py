import sys
import random
from typing import Tuple, Iterable
from sting.data import Feature, FeatureType
import numpy as np

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()
    n_zeros = y.size - n_ones
    return n_zeros, n_ones


def entropy(y: np.ndarray) -> float:
    """
    Calculates the entropy of a distribution from an array of values.

    Args:
        y (np.ndarray): an array of values

    Returns:
        float: the entropy of the values in y
    """

    n_entries = np.shape(y)[0]

    # count the number of occurences of each unique value
    _, counts = np.unique(y, return_counts=True)
    # calculate the probability density of each count
    densities = counts/n_entries
    # calculate the entropy from the observed probability density
    entropy = -np.sum(densities * np.log2(densities))

    return entropy


def binary_entropy(p: float) -> float:
    """
    Calculates the binary entropy of a bernoulli distribution

    Args:
        p (float): probability of the bernoulli distribution

    Returns:
        float: binary entropy of the bernoulli distribution on p
    """

    if p < sys.float_info.epsilon or abs(1-p) < sys.float_info.epsilon:
        return 0
    entropy = (
        -p * np.log2(p) - (1-p) * np.log2(1-p)
    )

    return entropy


def info_gain(y, x: np.ndarray) -> float:
    """
    Args:
        y: entropy prior to partitioning
        x: array post partitioning

    Returns: the information gain of the array

    """
    return (y - entropy(x))


def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
        ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    # merge labels to the features matrix
    samples = np.concatenate((X, np.array([y]).T), axis=1)
    n_samples = samples.shape[0]

    if stratified:
        # partition samples to positive and negative samples
        negative_samples = samples[y == 0]
        positive_samples = samples[y == 1]
        np.random.shuffle(negative_samples)
        np.random.shuffle(positive_samples)

        negative_datasets = cv_split(
            negative_samples[:, :-1],
            negative_samples[:, -1],
            folds=folds,
            stratified=False)
        positive_datasets = cv_split(
            positive_samples[:, :-1],
            positive_samples[:, -1],
            folds=folds,
            stratified=False)

        datasets = []
        for (nd, pd) in zip(negative_datasets, positive_datasets):
            nX_train, ny_train, nX_test, ny_test = nd
            pX_train, py_train, pX_test, py_test = pd

            datasets.append((
                np.concatenate((nX_train, pX_train)),
                np.concatenate((ny_train, py_train)),
                np.concatenate((nX_test, pX_test)),
                np.concatenate((ny_test, py_test)),
            ))

        return tuple(datasets)
    else:
        # shuffle samples
        np.random.shuffle(samples)

        # generate folds
        fold_size = int(np.ceil(n_samples/folds))

        datasets = []
        for i in range(folds):
            a = i * fold_size
            b = min(a+fold_size, n_samples)
            training_set = np.concatenate((samples[:a], samples[b:]))
            test_set = samples[a:b]
            datasets.append((
                training_set[:, :-1],
                training_set[:, -1],
                test_set[:, :-1],
                test_set[:, -1]
            ))

        return tuple(datasets)


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """
    y, y_hat = np.array(y), np.array(y_hat)
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same size/shape!')
    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    """

    Args:
        y: True labels
        y_hat:Predicted Labels

    Returns: Precision (TP/(TP + FP)
    """
    y, y_hat = np.array(y), np.array(y_hat)
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same size/shape!')

    positive_predictions = (y_hat == 1).sum()
    true_positives = (y * y_hat).sum()
    return true_positives / positive_predictions


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    """

    Args:
        y: True Labels
        y_hat: Predicted Labels

    Returns: recall TP/TP + FN

    """
    y, y_hat = np.array(y), np.array(y_hat)
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same size/shape!')

    true_positives = (y * y_hat).sum()
    all_positives = (y == 1).sum()
    return true_positives / all_positives


def fp_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """

    Args:
        y: True Labels
        y_hat: Predicted Labels

    Returns: recall TP/TP + FN

    """
    y, y_hat = np.array(y), np.array(y_hat)
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same size/shape!')

    false_positives = ((y != y_hat) & (y == 0)).sum()
    all_real_negatives = (y == 0).sum()
    return false_positives / all_real_negatives


def cut_points(x: np.ndarray, y: np.ndarray, return_counts=False) -> np.ndarray:
    """Generates cut points for a continuous feature

    Args:
        x (np.ndarray): continuous feature values
        y (np.ndarray): class labels

    Returns:
        np.ndarray: an array of cut points
    """

    # secondary sort by y
    pre_sort = np.argsort(y)
    x = x[pre_sort]
    y = y[pre_sort]

    # sort x and y by values of x
    sort_order = np.argsort(x)
    x = x[sort_order]
    y = y[sort_order]

    difs = np.nonzero(y[:-1] - y[1:])[0]  # nonzero difference between consecutive labels
    cuts = (x[difs + 1] + x[difs]) / 2  # take the average of consecutive feature values with different labels
    _, unique_cut_i = np.unique(cuts, return_index=True)
    cuts = cuts[unique_cut_i]

    if return_counts:
        x_difs = np.where(x[1:] - x[:-1] == 0, 0, 1)
        counts = np.arange(x.size)
        pos_counts = np.zeros(x.size)
        # dp iteration, update counts for same x values
        # example: x = [1, 2, 2, 2, 3, 3, 4, 4, 4] should have counts = [0, 1, 1, 1, 4, 4, 6, 6, 6]
        #        x_difs = [1, 0, 0, 1, 0, 1, 0, 0]
        for i in range(1, x.size):
            if x_difs[i-1] == 0:
                counts[i] = counts[i-1]
                pos_counts[i] = pos_counts[i-1]
            else:
                pos_counts[i] = np.sum(y[:i])

        cut_counts = counts[difs+1]
        cut_counts = cut_counts[unique_cut_i]

        pos_counts = pos_counts[difs+1]
        pos_counts = pos_counts[unique_cut_i]

        return cuts, cut_counts, pos_counts
    else:
        return cuts


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
     Args:
        y: true values, in order corresponding to confidence values
        p_y_hat: list of probabilities in descending order

    Returns: The recall, precision at each confidence point
    """
    point_list = []
    for i in range(len(p_y_hat)):
        if p_y_hat[i] != p_y_hat[i-1]:
            y_hat = np.where((np.float32(p_y_hat) >= np.float32(p_y_hat[i])), 1, 0)
            point_list.append([fp_rate(y, y_hat), recall(y, y_hat)])
    return point_list


def split_bins(continuous: Feature, feature_values: np.ndarray, bin_count: int) -> Tuple[Feature, np.ndarray]:
    """Discretizes a continuous feature by partitioning into bins of equal intervals.

    Args:
        continuous (Feature): Continuous feature to discretize
        feature_values (np.ndarray): Values of the continuous feature
        bin_number (int): Number of bins to partition to

    Returns:
        Tuple[Feature, np.ndarray]: [description]
    """
    # calculate range of feature values
    max_value = np.amax(feature_values)
    min_value = np.amin(feature_values)
    feature_range = max_value - min_value
    bin_size = feature_range / bin_count

    # discretize continuous values into bins
    discretized_values = np.floor((feature_values - min_value) / bin_size)+1  # nominal values start at 1
    discretized_values[feature_values == max_value] = bin_count  # put maximum value in the last bin

    discretized_feature = Feature(
        continuous.name,
        FeatureType.NOMINAL,
        nominal_values=[f"range[{min_value + bin_size*i}, {min_value + bin_size*(i+1)}]" for i in range(bin_count)],
        )
    return discretized_feature, discretized_values


def encode_values(nominal: Feature, feature_values: np.ndarray) -> np.ndarray:
    """Encodes a nominal feature by mapping each value to a number from 1...k

    Args:
        nominal (Feature): Nominal feature to discretize
        feature_values (np.ndarray): Values of the nominal feature

    Returns:
        Tuple[Feature, np.ndarray]: [description]
    """
    unique_array = np.unique(feature_values)
    encoding_dictionary = {}
    j = 1
    # Created a dicitionary to map unique values with corresponding int 1...k
    for i in unique_array:
        encoding_dictionary[i] = j
        j = j+1

    # Creating a new feature values array with encoded values
    new_feature_values = np.array([encoding_dictionary[k] for k in feature_values])
    return new_feature_values


def sigmoid(x):
    """

    Args:
        x: the function to take the sigmoid of

    Returns: the sigmoid function of x

    """
    return 1/(1 + np.exp(-x))

def sort_confidences(y: np.ndarray, p_y_hat: np.ndarray):
    # P_y_hat is the confidence label
    sorted_confidenceZip = zip(y, p_y_hat)
    sorted_confidence = np.array(list(sorted_confidenceZip))
    sorted_confidence = sorted_confidence[sorted_confidence[:, 1].argsort()[::-1]]
    return sorted_confidence

def area_under_roc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    """
     Args:
        y: true values
        p_y_hat: list of probabilities in descending order

    Returns: The area under the ROC curve given the provided confidence estimates
    """
    roc_curve_points = roc_curve_pairs(y, p_y_hat)
    last_point = (0, 0)
    area_sum = 0.0
    for each_point in roc_curve_points:
        area_sum += 0.5*(each_point[0]-last_point[0])*(last_point[1]+each_point[1])
        last_point = each_point
    area_sum += 0.5*(1-last_point[0])*(last_point[1]+1)
    return area_sum

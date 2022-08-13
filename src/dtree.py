import argparse
import os.path

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

import util
from split_criteria import (
    SplitCriterion,
    split_by_info_gain, split_by_gain_ratio, stochastic_split
)


# Morgan Gillaspie, Lexi Scott, Ishaan Gupta, Harry Kwon
class DecisionTree(Classifier):
    def __init__(
            self,
            schema: List[Feature],
            split_criterion: SplitCriterion = split_by_info_gain,
            ):
        self._schema = schema
        self._majority_label = 0
        self.split_attribute = None  # index of split attribute
        self.split_boundary = None  # value of continuous attribute split boundary
        self._successors = {}

        # dependency injection on split criterion
        self.split_criterion = split_criterion
        self.size = 1
        self.depth = 1

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None, depth=0, random=False) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """
        # generate majority label at this node
        n_zero, n_one = util.count_label_occurrences(y)

        if n_one > n_zero:
            self._majority_label = 1
        else:
            self._majority_label = 0

        # skip partitioning if depth limit reached or pure node
        if depth == 1 or min(n_zero, n_one) == 0:
            return

        if (random) and (np.random.randint(100) < 30):
            self.split_attribute, self.split_boundary, partition = stochastic_split(self.schema, X, y)
        else:
            self.split_attribute, self.split_boundary, partition = self.split_criterion(self.schema, X, y)

        if self.split_attribute is None:  # no attribute can partition the dataset
            return

        self.size += 2

        # partition examples by the split attribute and generate a DecisionTree for each subset
        for key, (p_X, p_y) in partition.items():
            # create a successor DTree
            successor_tree = DecisionTree(self._schema)

            # fit the successor Dtree on the subset of examples
            successor_tree.fit(p_X, p_y, depth=depth - 1)

            self.depth = max(successor_tree.depth+1, self.depth)

            self.size += successor_tree.size

            # append the successor tree to this tree's successors
            self._successors[key] = successor_tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """
        if self.split_attribute is None:
            # no decision boundary at this node / is a leaf node
            # Returns either all 1s or all 0s, depending on _majority_label.
            return np.ones(X.shape[0], dtype=int) * self._majority_label
        else:
            split_values = X[:, self.split_attribute]
            if self.schema[self.split_attribute].ftype == FeatureType.CONTINUOUS:
                split_values = np.where(split_values > self.split_boundary, 1, 0)
            y = np.zeros(X.shape[0])

            for k in self._successors.keys():
                i = (split_values == k)
                y[i] = self._successors[k].predict(X[i])
            return y

    @property
    def schema(self):
        """
        Returns: The dataset schema as a dictionary mapping index to Feature
        """
        return self._schema


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size: ', dtree.size)
    print('Maximum Depth: ', dtree.depth)
    if dtree.split_attribute is not None:
        first_feature = dtree.schema[dtree.split_attribute].name
    else:
        first_feature = None
    print('First Feature: ', first_feature)


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True, random: bool = False):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    split_criterion = split_by_info_gain if information_gain else split_by_gain_ratio

    if(random):
        best_accuracy = 0
        for i in range(5):
            print("beginning tree number " + str(i))
            for X_train, y_train, X_test, y_test in datasets:
                new_tree = DecisionTree(schema, split_criterion)
                new_tree.fit(X_train, y_train, depth=tree_depth_limit, random=True)
                evaluate_and_print_metrics(new_tree, X_test, y_test)
            y_hat = new_tree.predict(X_test)
            acc = util.accuracy(y_test, y_hat)
            if acc > best_accuracy:
                best_accuracy = acc
                best_tree = new_tree
        print("Best tree found:")
        evaluate_and_print_metrics(best_tree, X_test, y_test)

    else:
        for X_train, y_train, X_test, y_test in datasets:
            # import time
            # __start_time = time.perf_counter()

            decision_tree = DecisionTree(schema, split_criterion)
            decision_tree.fit(X_train, y_train, depth=tree_depth_limit)
            # print(f"Training completed in {time.perf_counter() - __start_time}")

            evaluate_and_print_metrics(decision_tree, X_test, y_test)


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')

    # This there to enable the random choice algorithm for our experiment
    parser.add_argument('--randomize', dest='random', action='store_true',
                        help='Enables the random selection feature')
    parser.set_defaults(cv=True, gain_ratio=False, random=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    random = args.random
    use_information_gain = not args.gain_ratio
    use_gain_ratio = args.gain_ratio
    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain, random)

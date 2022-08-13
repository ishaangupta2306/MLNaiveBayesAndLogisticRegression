from typing import List, Dict, Tuple, Callable
import warnings
import numpy as np

from sting.data import Feature, FeatureType
import util
from util import binary_entropy

# define types
FeatureIndex = int
SplitBoundary = float
Partition = Dict[int, Tuple[np.ndarray, np.ndarray]]
SplitCriterion = Callable[
    [List[Feature], np.ndarray, np.ndarray],
    Tuple[FeatureIndex, SplitBoundary, Partition]
]


def split_by_gain_ratio(
        schema: List[Feature],
        X: np.ndarray,
        y: np.ndarray
        ) -> Tuple[FeatureIndex, SplitBoundary, Partition]:
    """Returns the feature with maximum gain ratio when partitioned by
    and the partition by that feature

    Args:
        schema (List[Feature]): dataset features
        X (np.ndarray): features of each examples
        y (np.ndarray): class labels

    Returns:
        split_feature (int): index of feature whose partition maximizes information gain on X, y
        split_boundary (float): split boundary for continuous features
        max_partition (Tuple[np.ndarray, np.ndarray]): a partition of X, y by max_feature. see split_criteria.generate_partitions() for details
    """
    prior_entropy = util.entropy(y)
    prior_pos = y.sum()
    max_gain_ratio = np.NINF

    def gain_ratio(partitions, feature_index):
        # calculate information gain of a partition
        partition_entropy = sum((p_y.size * util.entropy(p_y)) for _, p_y in partitions.values())
        partition_entropy /= y.size
        information_gain = (prior_entropy - partition_entropy)
        # calculate entropy of the feature
        entropy = util.entropy(X[:, feature_index])
        return information_gain / entropy

    def binary_gain_ratio(count, pos_count):
        s1 = count
        s2 = y.size - s1
        p1 = pos_count
        p2 = prior_pos - p1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            partition_entropy = (
                s1 * binary_entropy(p1/s1) + s2 * binary_entropy(p2/s2)
            ) / y.size
        ig = prior_entropy - partition_entropy
        return ig / binary_entropy(s1/y.size)

    for i in range(len(schema)):
        if schema[i].ftype in [FeatureType.BINARY, FeatureType.NOMINAL]:
            parts = generate_partitions(schema, X, y, i)
            ratio = gain_ratio(parts, i)
            if ratio > max_gain_ratio and len(parts) > 1:  # skip boundaries that do not partition proper subsets
                max_gain_ratio, max_attr, max_part = ratio, (i, None), parts

        elif schema[i].ftype == FeatureType.CONTINUOUS:
            cut_points, cut_counts, pos_counts = util.cut_points(X[:, i], y, return_counts=True)

            for ci in range(cut_points.size):
                gr = binary_gain_ratio(cut_counts[ci], pos_counts[ci])
                if gr > max_gain_ratio and cut_counts[ci] > 0:  # skip boundaries that do not partition proper subsets
                    max_gain_ratio = gr
                    max_attr = (i, cut_points[ci])
                    max_part = generate_partitions(schema, X, y, i, cut_points[ci])
    return *max_attr, max_part


def split_by_info_gain(
        schema: List[Feature],
        X: np.ndarray,
        y: np.ndarray
        ) -> Tuple[FeatureIndex, SplitBoundary, Partition]:
    """Returns the feature with maximum information gain when partitioned by
    and the partition by that feature

    Args:
        schema (List[Feature]): dataset features
        X (np.ndarray): features of each examples
        y (np.ndarray): class labels

    Returns:
        split_feature (int): index of feature whose partition maximizes information gain on X, y
        split_boundary (float): split boundary for continuous features
        max_partition (Tuple[np.ndarray, np.ndarray]): a partition of X, y by max_feature. see split_criteria.generate_partitions() for details
    """
    prior_entropy = util.entropy(y)
    prior_pos = y.sum()

    def info_gain(partitions):
        # calculate information gain of a partition
        partition_entropy = sum((p_y.size * util.entropy(p_y)) for _, p_y in partitions.values())
        partition_entropy /= y.size
        return prior_entropy - partition_entropy

    def binary_info_gain(count, pos_count):
        s1 = count
        s2 = y.size - s1
        p1 = pos_count
        p2 = prior_pos - p1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            partition_entropy = (
                s1 * binary_entropy(p1/s1) + s2 * binary_entropy(p2/s2)
            ) / y.size
        return prior_entropy - partition_entropy

    max_ig = np.NINF
    max_attr = (None, None)
    max_part = None
    # iterate through all features and find the feature and boundary the maximizes information gain
    for i in range(len(schema)):
        if schema[i].ftype == FeatureType.CONTINUOUS:
            cut_points, cut_counts, pos_counts = util.cut_points(X[:, i], y, return_counts=True)

            for ci in range(cut_points.size):
                ig = binary_info_gain(cut_counts[ci], pos_counts[ci])
                if ig > max_ig and cut_counts[ci] > 0:  # skip boundaries that do not partition proper subsets
                    max_ig = ig
                    max_attr = (i, cut_points[ci])
                    max_part = generate_partitions(schema, X, y, i, cut_points[ci])

        elif schema[i].ftype in [FeatureType.BINARY, FeatureType.NOMINAL]:
            parts = generate_partitions(schema, X, y, i)
            ig = info_gain(parts)
            if ig > max_ig and len(parts) > 1:  # skip boundaries that do not partition proper subsets
                max_ig, max_attr, max_part = ig, (i, None), parts
    return *max_attr, max_part


def stochastic_split(
        schema: List[Feature],
        X: np.ndarray,
        y: np.ndarray) -> Tuple[FeatureIndex, SplitBoundary, Partition]:
    """Selects a random feature from the schema as a split attribute.
    Select a random cut point (between different class labels) for continuous attributes.

    Args:
        schema (List[Feature]): dataset schema
        X (np.ndarray): features of each example
        y (np.ndarray): class labels

    Returns:
        split_feature (int): index of feature
        split_boundary (float): split boundary for continuous features
        max_partition (Tuple[np.ndarray, np.ndarray]): a partition of X, y by split_feature. see split_criteria.generate_partitions() for details
    """
    # Randomly picking up a Feature
    randomFeatureIndex = np.random.randint(0, len(schema))

    if schema[randomFeatureIndex].ftype == FeatureType.CONTINUOUS:
        cut_points_array = util.cut_points(X[:, randomFeatureIndex], y)
        cut_points_array = np.unique(cut_points_array)
        # Randomly picking up a Cut point
        randomCutPointArrayIndex = np.random.randint(0, len(cut_points_array))
        parts = generate_partitions(schema, X, y, randomFeatureIndex, cut_points_array[randomCutPointArrayIndex])
        return randomFeatureIndex,  cut_points_array[randomCutPointArrayIndex], parts

    else:
        parts = generate_partitions(schema, X, y, randomFeatureIndex)
        return randomFeatureIndex, None, parts


def generate_partitions(
        schema: List[Feature],
        X: np.ndarray,
        y: np.ndarray,
        split_feature: int,
        split_boundary: float = None
        ) -> Partition:
    """Partitions examples by a feature and boundary if continous

    Args:
        schema (List[Feature]): dataset schema
        X (np.ndarray): features of examples
        y (np.ndarray): class labels
        split_feature (int): index of the feature of partition by
        split_boundary (float, optional): split boundary if split_featuare is continuous. Defaults to None.

    Raises:
        ValueError: if no split_boundary is passed with a continuous feature

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray]]: a dictionary of (X, y) subsets in partition.
            Keys are the feature value in the subset.
            For continuous features, keys are 0 for examples less than, 1 for more than the boundary.
    """
    split_values = X[:, split_feature]
    if schema[split_feature].ftype == FeatureType.CONTINUOUS:
        try:
            split_values = np.where(split_values > split_boundary, 1, 0)
        except ValueError:
            raise ValueError("Continuous features must be partitioned at a split boundary")

    partition_keys = np.unique(split_values)
    partition = {}
    for key in partition_keys:
        i = (split_values == key)  # indexer for elements in this subset
        partition[key] = (X[i], y[i])

    return partition

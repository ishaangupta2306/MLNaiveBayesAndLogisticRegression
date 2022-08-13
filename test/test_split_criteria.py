import unittest
import numpy as np

from sting.data import Feature, FeatureType
from src.split_criteria import generate_partitions, split_by_info_gain, split_by_gain_ratio, stochastic_split


class TestSplitByInfoGain(unittest.TestCase):
    def test_split_by_info_gain(self):
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature(
                "nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        X = np.array([
            [0, 0, 1, 0],
            [1, 0, 2, 0.1],
            [0, 0, 3, 0.2],
            [1, 1, 1, 0.3],
            [0, 1, 2, 0.4],
            [1, 0, 3, 0.5],
            [0, 0, 1, 0.6],
            [1, 1, 2, 0.7],
            [0, 1, 3, 0.8],
            [1, 1, 1, 0.9]
        ])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        split_feature, split_boundary, partition = split_by_info_gain(schema, X, y)
        self.assertEqual(0, split_feature)
        self.assertEqual(2, len(partition))
        pX0, py0 = partition[0]
        self.assertTrue(np.allclose(X[[0, 2, 4, 6, 8]], pX0))
        self.assertTrue(np.array_equal(y[[0, 2, 4, 6, 8]], py0))
        pX1, py1 = partition[1]
        self.assertTrue(np.allclose(X[[1, 3, 5, 7, 9]], pX1))
        self.assertTrue(np.array_equal(y[[1, 3, 5, 7, 9]], py1))

        X = np.array([
            [0, 0, 1, 0],
            [1, 0, 2, 0.1],
            [0, 0, 3, 0.2],
            [1, 1, 1, 0.3],
            [0, 1, 2, 0.4],
            [1, 0, 3, 0.5],
            [0, 0, 1, 0.6],
            [1, 1, 2, 0.7],
            [0, 1, 3, 0.8],
            [1, 1, 1, 0.9]
        ])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        split_feature, split_boundary, partition = split_by_info_gain(schema, X, y)
        self.assertEqual(3, split_feature)
        self.assertAlmostEqual(0.45, split_boundary)
        self.assertEqual(2, len(partition))
        pX0, py0 = partition[0]
        self.assertTrue(np.allclose(X[[0, 1, 2, 3, 4]], pX0))
        self.assertTrue(np.array_equal(y[[0, 1, 2, 3, 4]], py0))
        pX1, py1 = partition[1]
        self.assertTrue(np.allclose(X[[5, 6, 7, 8, 9]], pX1))
        self.assertTrue(np.array_equal(y[[5, 6, 7, 8, 9]], py1))


class TestSplitByGainRatio(unittest.TestCase):
    def test_split_by_gain_ratio(self):
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature(
                "nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        X = np.array([
            [0, 0, 1, 0],
            [1, 0, 2, 0.1],
            [0, 0, 3, 0.2],
            [1, 1, 1, 0.3],
            [0, 1, 2, 0.4],
            [1, 0, 3, 0.5],
            [0, 0, 1, 0.6],
        ])
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        split_feature, split_boundary, partition = split_by_gain_ratio(schema, X, y)
        self.assertEqual(0, split_feature)
        self.assertEqual(2, len(partition))


class StochasticSplit(unittest.TestCase):
    def test_stochastic_split(self):
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature(
                "nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("binary1", FeatureType.BINARY),
        ]
        X = np.array([
            [0, 0, 1, 0],
            [1, 0, 2, 1],
            [0, 0, 3, 1],
            [1, 1, 1, 0],
            [0, 1, 2, 0],
            [1, 0, 3, 1],
            [0, 0, 1, 1],
        ])
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        randomFeatureIndex, split_boundary, partition = stochastic_split(schema, X, y)
        self.assertEqual(randomFeatureIndex in range(0, 4, 1), True)


class TestPartition(unittest.TestCase):
    def test_binary(self):
        schema = [
            Feature("b1", FeatureType.BINARY),
            Feature("b2", FeatureType.BINARY),
            Feature("b3", FeatureType.BINARY),
        ]
        X = np.array([
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1]
        ])
        y = np.array([1, 2, 3])

        partitions = generate_partitions(schema, X, y, 1)

        self.assertListEqual([0, 1], list(partitions.keys()))

        X_0_expected = X[[2]]
        y_0_expected = y[[2]]
        X_0, y_0 = partitions[0]
        self.assertTrue(np.array_equal(X_0_expected, X_0))
        self.assertTrue(np.array_equal(y_0_expected, y_0))

        X_1_expected = X[[0, 1]]
        y_1_expected = y[[0, 1]]
        X_1, y_1 = partitions[1]
        self.assertTrue(np.array_equal(X_1_expected, X_1))
        self.assertTrue(np.array_equal(y_1_expected, y_1))

    def test_nominal(self):
        schema = [
            Feature(
                "n1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("b1", FeatureType.BINARY),
            Feature(
                "n2", FeatureType.NOMINAL,
                [f"n{x}" for x in range(6)]
            ),
        ]
        X = np.array([
            [0, 1, 1],
            [1, 1, 2],
            [2, 0, 3],
            [0, 1, 4],
            [1, 1, 5],
            [2, 0, 6]
        ])
        y = np.arange(6)

        partitions = generate_partitions(schema, X, y, 0)

        self.assertListEqual([0, 1, 2], list(partitions.keys()))

        X_0_expected = X[[0, 3]]
        y_0_expected = y[[0, 3]]
        X_0, y_0 = partitions[0]
        self.assertTrue(np.array_equal(X_0_expected, X_0))
        self.assertTrue(np.array_equal(y_0_expected, y_0))

        X_1_expected = X[[1, 4]]
        y_1_expected = y[[1, 4]]
        X_1, y_1 = partitions[1]
        self.assertTrue(np.array_equal(X_1_expected, X_1))
        self.assertTrue(np.array_equal(y_1_expected, y_1))

        X_2_expected = X[[2, 5]]
        y_2_expected = y[[2, 5]]
        X_2, y_2 = partitions[2]
        self.assertTrue(np.array_equal(X_2_expected, X_2))
        self.assertTrue(np.array_equal(y_2_expected, y_2))

    def test_continuous(self):
        schema = [
            Feature("c1", FeatureType.CONTINUOUS),
            Feature("b1", FeatureType.BINARY),
            Feature(
                "n1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(6)]
            ),
        ]
        X = np.array([
            [0.0, 1, 1],
            [0.6, 1, 2],
            [0.3, 0, 3],
            [0.5, 1, 4],
            [0.2, 1, 5],
            [0.9, 0, 6]
        ])
        y = np.arange(6)

        partitions = generate_partitions(schema, X, y, 0, 0.4)

        self.assertListEqual([0, 1], list(partitions.keys()))

        X_0_expected = X[[0, 2, 4]]
        y_0_expected = y[[0, 2, 4]]
        X_0, y_0 = partitions[0]
        self.assertTrue(np.array_equal(X_0_expected, X_0))
        self.assertTrue(np.array_equal(y_0_expected, y_0))

        X_1_expected = X[[1, 3, 5]]
        y_1_expected = y[[1, 3, 5]]
        X_1, y_1 = partitions[1]
        self.assertTrue(np.array_equal(X_1_expected, X_1))
        self.assertTrue(np.array_equal(y_1_expected, y_1))

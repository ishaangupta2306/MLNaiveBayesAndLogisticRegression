import unittest
import numpy as np

from sting.data import Feature, FeatureType
from src.dtree import DecisionTree


class TestDecisionTree(unittest.TestCase):
    def test_fit(self):
        # @TODO test for correctness of subtrees?
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature("nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        decision_tree = DecisionTree(schema)

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
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        # test termination
        decision_tree.fit(X, y, depth=0)

    def test_predict(self):
        # @TODO better tests. too tired to generate examples right now
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature("nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        decision_tree = DecisionTree(schema)

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
        # test termination
        decision_tree.fit(X, y, depth=0)
        predictions = decision_tree.predict(np.array([
            [1, 0, 2, 0.1],
            [0, 0, 1, 0.6],
            [1, 1, 1, 0.47],
            [1, 1, 1, 0.43],
        ]))
        predictions_expected = np.array([0, 1, 1, 0])
        print("Predictions: ", predictions)
        self.assertTrue(np.allclose(predictions, predictions_expected))

    def test_maxDepth_bounded(self):
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature("nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        decision_tree = DecisionTree(schema)

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
        # test termination
        decision_tree.fit(X, y, depth=1)
        self.assertTrue(decision_tree.depth == 1)

    def test_maxDepth_unbounded(self):
        schema = [
            Feature("binary1", FeatureType.BINARY),
            Feature("binary2", FeatureType.BINARY),
            Feature("nominal1", FeatureType.NOMINAL,
                [f"n{x}" for x in range(3)]
            ),
            Feature("continuous", FeatureType.CONTINUOUS),
        ]
        decision_tree = DecisionTree(schema)

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
        # test termination
        decision_tree.fit(X, y, depth=0)
        self.assertEqual(decision_tree.depth, 2)

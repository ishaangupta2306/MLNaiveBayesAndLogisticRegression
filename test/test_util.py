import unittest
import numpy as np

from sting.data import Feature, FeatureType
from src.util import area_under_roc, accuracy, precision, recall, entropy, cv_split, info_gain, fp_rate, roc_curve_pairs, cut_points, split_bins, encode_values


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(1., accuracy([1, 1, 1, 1], [1, 1, 1, 1]))
        self.assertEqual(0.5, accuracy(np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1])))

    def test_precision(self):
        self.assertEqual(1., precision([1, 1, 1, 1], [1, 1, 1, 1]))
        self.assertEqual(0.5, precision(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])))

    def test_recall(self):
        self.assertEqual(1., recall([1, 1, 1, 1], [1, 1, 1, 1]))
        self.assertEqual(0.5, recall(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])))

    def test_entropy(self):
        self.assertEqual(1,  entropy(np.array([0, 0, 1, 1])))

    def test_info_gain(self):
        self.assertEqual(0,  info_gain(1,  np.array([0, 0, 1, 1])))

    def test_fp_rate(self):
        self.assertEqual(0.5, fp_rate([1, 0, 1, 0], [1, 1, 1, 0]))

    def test_roc_curve(self):
        self.assertEqual(
            [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [1.0, 1.0]],
            roc_curve_pairs([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.3]))

    def test_area_under_roc(self):
        self.assertEqual(
            0.75,
            area_under_roc([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.3]))


class TestEntropy(unittest.TestCase):
    def test_single(self):
        t1 = np.zeros(10,)
        self.assertAlmostEqual(entropy(t1), 0)

        t2 = np.ones(10,)
        self.assertAlmostEqual(entropy(t2), 0)

        t3 = np.full(10, fill_value=17.01)
        self.assertAlmostEqual(entropy(t3), 0)

    def test_binary(self):
        t1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(entropy(t1), 1)

        t2 = np.array([1, 1, 1, 0])
        self.assertAlmostEqual(entropy(t2), 0.8112781244591328)

    def test_nominal(self):
        t1 = np.arange(10)
        self.assertAlmostEqual(entropy(t1), 3.32192809488736)


class TestCVSplit(unittest.TestCase):
    def test_basic(self):
        samples = 9
        folds = 3
        X = np.arange(samples)
        X = np.array([X]).T
        y = np.zeros(samples)

        datasets = cv_split(X, y, folds, False)

        # 5 folds should be generated
        self.assertEqual(len(datasets), folds)

        for (X_train, y_train, X_test, y_test) in datasets:
            # the training set should contain 6 samples (4 folds)
            self.assertEqual(X_train.shape, (6, 1))
            self.assertEqual(y_train.shape, (6,))

            # the test set should contain 3 samples (1 fold)
            self.assertEqual(X_test.shape, (3, 1))
            self.assertEqual(y_test.shape, (3,))

            # no sample in the training set should be in the test set
            # union of training set and test set should return original samples
            c = np.concatenate((X_train.flatten(), X_test.flatten()))
            self.assertTrue(np.array_equal(np.sort(c), X.flatten()))

    def test_unstratified(self):
        features = 5
        samples = 20
        folds = 5
        X = np.arange(features * samples).reshape((samples, features))
        y = np.zeros(samples,)

        datasets = cv_split(X, y, folds, False)

        # 5 folds should be generated
        self.assertEqual(len(datasets), folds)

        for (X_train, y_train, X_test, y_test) in datasets:
            # the training set should contain 16 samples (4 folds)
            self.assertEqual(X_train.shape, (16, 5))
            self.assertEqual(y_train.shape, (16,))

            # the test set should contain 4 samples (1 fold)
            self.assertEqual(X_test.shape, (4, 5))
            self.assertEqual(y_test.shape, (4,))

            # no sample in the training set should be in the test set
            # (flatten both )
            c = np.concatenate((X_train.flatten(), X_test.flatten()))
            self.assertTrue(np.array_equal(np.sort(c), X.flatten()))

    def test_stratified(self):
        features = 5
        samples = 20
        folds = 5
        X = np.arange(features * samples).reshape((samples, features))
        y = np.array([0, 1, 1, 1] * 5)  # 3/4 positive labels, 1/4 negative

        datasets = cv_split(X, y, folds, stratified=True)

        # 5 folds should be generated
        self.assertEqual(len(datasets), folds)

        for (X_train, y_train, X_test, y_test) in datasets:
            # the training set should contain 16 samples (4 folds)
            self.assertEqual(X_train.shape, (16, 5))
            self.assertEqual(y_train.shape, (16,))

            # each training set should have 12 positive, 4 negative samples
            self.assertEqual(np.sum(y_train == 1), 12)
            self.assertEqual(np.sum(y_train == 0), 4)

            # the test set should contain 4 samples (1 fold)
            self.assertEqual(X_test.shape, (4, 5))
            self.assertEqual(y_test.shape, (4,))

            # each test set should have 3 positive, 1 negative samples
            self.assertEqual(np.sum(y_test == 1), 3)
            self.assertEqual(np.sum(y_test == 0), 1)

            # no sample in the training set should be in the test set
            # (flatten both )
            c = np.concatenate((X_train.flatten(), X_test.flatten()))
            self.assertTrue(np.array_equal(np.sort(c), X.flatten()))


class TestCutPoints(unittest.TestCase):
    def test_cut_points(self):
        x = np.arange(10) / 10
        y = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1])
        expected_cuts = np.array([0.05, 0.45, 0.55, 0.75, 0.85])
        expected_counts = np.array([1, 5, 6, 8, 9])
        expected_pos_counts = np.array([0, 4, 4, 6, 6])

        # shuffle samples
        np.random.seed(12345)
        s = np.arange(10)
        np.random.shuffle(s)
        x = x[s]
        y = y[s]

        cuts, counts, pos_counts = cut_points(x, y, return_counts=True)

        self.assertTrue(np.allclose(cuts, expected_cuts))
        self.assertTrue(np.allclose(counts, expected_counts))
        self.assertTrue(np.allclose(pos_counts, expected_pos_counts))


class TestNaiveBayes(unittest.TestCase):
    def test_split_bins(self):
        continuous_feature = Feature("continuous feature", FeatureType.CONTINUOUS)
        values_input = [0, 1, 2, 3, 5, 6, 7.8, 8, 10]
        bin_number = 5
        new_feature, new_values = split_bins(continuous_feature, values_input, bin_number)
        self.assertEqual(list(new_values), [1.0, 1.0,  2.0,  2.0,  3.0,  4.0,  4.0,  5.0, 5.0])
        self.assertEqual(new_feature, Feature("continuous feature", FeatureType.NOMINAL, ["0.0", "2.0", "4.0", "6.0", "8.0"]))

    def test_encode_values(self):
        nominal_feature = Feature("nominal feature", FeatureType.NOMINAL, ['red', 'blue', 'green'])
        values_input = ['red', 'green', 'green', 'red', 'blue', 'green', 'blue']
        new_values = encode_values(nominal_feature, values_input)
        self.assertEqual(list(new_values), [3, 2, 2, 3, 1, 2, 1])


if __name__ == '__main__':
    unittest.main()

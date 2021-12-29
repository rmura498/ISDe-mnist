import unittest
from nmc import NMC
import numpy as np


class TestNMC(unittest.TestCase):

    def setUp(self):
        n_samples = 100
        n_features = 20
        self.x = np.zeros(shape=(n_samples, n_features))
        self.y = np.ones(shape=(n_samples,))
        self.clf = NMC()

    def test_init(self):
        """Check if centroids are None right after creation"""
        self.assertTrue(self.clf.centroids is None)

    def test_robust_estimation(self):
        self.assertRaises(TypeError, setattr, self.clf, 'robust_estimation', 'nonsonoboleano')


    def test_fit(self):
        expected_centroid_shape = (1, self.x.shape[1])
        self.clf.fit(self.x, self.y)
        self.assertEqual(self.clf.centroids.shape, expected_centroid_shape)

    def test_predict(self):
        pass

import unittest
from nmc import NMC
import numpy as np



class TestNMC(unittest.TestCase):

    def SetUp(self):
        n_samples = 100
        n_features = 20
        self.x = np.zeros(shape=(n_samples, n_features))
        self.y = np.ones(shape=(n_samples,))

    def test_fit(self):
        pass

    def test_predict(self):
        pass

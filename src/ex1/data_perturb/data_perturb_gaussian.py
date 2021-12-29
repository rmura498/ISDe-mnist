from data_perturb import CDataPerturb
import numpy as np


class CDataPerturbGaussian(CDataPerturb):

    def __init__(self, sigma=100, min_value=0, max_value=255):
        super().__init__()
        self.sigma = sigma
        self.min_value = min_value
        self.max_value = max_value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = int(value)

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        self._max_value = int(value)

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = int(value)

    def data_perturbation(self, x):
        if x.size != x.shape[0]:
            raise TypeError("x is not flattened")

        xp = x.copy().ravel()

        xp = xp + self.sigma * np.random.randn(xp.size)
        xp[xp < self.min_value] = self.min_value
        xp[xp > self.max_value] = self.max_value

        return xp
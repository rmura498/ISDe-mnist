import numpy as np
from sklearn.metrics import pairwise_distances


class NMC:
    """Class implementing the Nearest Mean Centroid (NMC)
    classification algorithm."""

    def __init__(self, robust_estimation=False):
        """
        Create the NMC object.
        :param robust_estimation: True, use medians to estimate centroids,
            mean otherwise
        """
        self.robust_estimation = robust_estimation
        self._centroids = None  # init centroids

    @property
    def centroids(self):
        return self._centroids

    @property
    def robust_estimation(self):
        return self._robust_estimation

    @robust_estimation.setter
    def robust_estimation(self, value):
        if not isinstance(value, bool):
            raise TypeError("value is not bool!")
        self._robust_estimation = value

    def fit(self, x_tr, y_tr):
        """Fit the model to the data (estimating centroids)"""
        n_classes = np.unique(y_tr).size
        n_features = x_tr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(n_classes):
            # extract only images of 0 from x_tr
            xk = x_tr[y_tr == k, :]
            if self.robust_estimation is False:
                self._centroids[k, :] = np.mean(xk, axis=0)
            else:
                self._centroids[k, :] = np.median(xk, axis=0)
        return self

    def predict(self, x_ts):
        dist = pairwise_distances(x_ts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
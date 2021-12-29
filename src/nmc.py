import numpy as np
from sklearn.metrics import pairwise_distances


class NMC():
    """Class implementing the Nearest Mean Centroid (NMC) classification algorithm"""

    def __init__(self, avg=True):
        self.avg = avg  # avg True means use mean, False use median
        self._centroids = None

    @property
    def avg(self):
        return self._avg

    @avg.setter
    def avg(self, value):
        self._avg = bool(value)

    @property
    def centroids(self):
        return self._centroids

    def fit(self, x_tr, y_tr):
        """Fit the model to the data (estimating centroids)"""
        n_classes = np.unique(y_tr).size
        n_features = x_tr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))

        for k in range(n_classes):
            # extract only images of 0 from x_tr
            xk = x_tr[y_tr == k, :]
            if self._avg is True:
                self._centroids[k, :] = np.mean(xk, axis=0)
            else:
                self._centroids[k, :] = np.median(xk, axis=0)

    def predict(self, x_ts):
        dist = pairwise_distances(x_ts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
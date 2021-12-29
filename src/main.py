from utils import load_mnist_data, split_data, plot_ten_digits, \
    load_mnist_data_openml
from nmc import NMC
import numpy as np

n_rep = 1

x, y = load_mnist_data()
# plot_ten_digits(x, y)

test_error = np.zeros(shape=(n_rep,))
for r in range(n_rep):
    print(r)
    x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=1000)
    clf = NMC(robust_estimation=True)
    clf.fit(x_tr, y_tr)
    plot_ten_digits(clf.centroids)
    ypred = clf.predict(x_ts)

    test_error[r] = (ypred != y_ts).mean()

print(test_error.mean(), test_error.std())
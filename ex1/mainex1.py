import numpy as np
import matplotlib.pyplot as plt

from utils import load_mnist_data, split_data, plot_ten_digits
from nmc import NMC

from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from data_perturb import CDataPerturbRandom, CDataPerturbGaussian


def robustness_test(clf, data_pert, param_name, param_values):
    """
    Running a robustness test on clf using the data_pert perturbation model.
    The test is run by setting param_name to different values (param_values).
    Parameters
    ----------
    clf :
        an object implementing fit and predict functions (sklearn interface)
    data_pert : ...
    param_name : ...
    param_values : ...
    Returns
    -------
    test_accuracies :
                      Accuracy values ...
    """
    test_accuracies = np.zeros(shape=param_values.shape)
    for i, k in enumerate(param_values):
        setattr(data_pert, param_name, k)  # data_pert.sigma = k
        xp = data_pert.perturb_dataset(x_ts)
        # plot_ten_digits(xp, y)
        # compute predicted labels on the perturbed ts
        y_pred = clf.predict(xp)
        # compute classification accuracy using y_pred
        clf_acc = np.mean(y_ts == y_pred)
        # print("Test accuracy(K=", k, "): ", int(clf_acc * 10000) / 100, "%")
        test_accuracies[i] = clf_acc
    return test_accuracies


x, y = load_mnist_data()

# implementing perturb_dataset(x) --> xp (perturbed dataset)
# initialize Xp
# loop over the rows of X, then at each iteration:
#    extract the given row,
#    apply the data_perturbation function
#    copy the result (perturbed image) in xp

# split MNITS data into 60% training and 40% test sets
n_tr = int(0.6 * x.shape[0])
print("Number of total samples: ", x.shape[0],
      "\nNumber of training samples: ", n_tr)

x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

param_values = np.array([0, 10, 20, 100, 200, 300, 400])

clf_list = [NMC(), SVC(kernel='linear')]
clf_names = ['NMC', 'SVM']

plt.figure(figsize=(10,5))

for i, clf in enumerate(clf_list):

    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    clf_acc = np.mean(y_ts == y_pred)
    print("Test accuracy: ", int(clf_acc * 10000) / 100, "%")

    test_accuracies = robustness_test(
        clf,  CDataPerturbRandom(), param_name='K', param_values=param_values)

    plt.subplot(1, 2, 1)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel('K')
    plt.ylabel('Test accuracy(K)')
    plt.legend()

    test_accuracies = robustness_test(
        clf,  CDataPerturbGaussian(), param_name='sigma', param_values=param_values)

    plt.subplot(1, 2, 2)
    plt.plot(param_values, test_accuracies, label=clf_names[i])
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'Test accuracy($\sigma$)')
    plt.legend()

plt.show()
from nmc import NMC
from utils import *

from data_perturb import CDataPerturbRandom, CDataPerturbGaussian


def robustness_test(clf, data_pert, param_name, param_values):
    test_accuracies = np.zeros(shape=param_values.shape)

    for i, k in enumerate(param_values):
        setattr(data_pert, param_name, k)
        xp = data_pert.perturb_dataset(x_ts)
        # plot_ten_digits(xp, y)
        y_pred = clf.predict(xp)
        clf_acc = np.mean(y_ts == y_pred)
        print("Test accuracy (K=", k, "):", (clf_acc * 1000) / 10, "%")
        test_accuracies[i] = clf_acc
    return test_accuracies


x, y = load_mnist_data()

# split MNITS data into training and test sets


n_tr = int(0.6 * x.shape[0])
print("Number of total samples: ", x.shape[0], "number of training samples: ", n_tr)

x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=n_tr)

clf = NMC()
clf.fit(x_tr, y_tr)

param_values = np.array([0, 10, 20, 50, 100, 200, 300, 400, 500])
data_pert_random = CDataPerturbRandom()
xp1 = data_pert_random.perturb_dataset(x)
plt.figure(figsize=(10, 5))
test_accuracies = robustness_test(clf, data_pert_random, param_name="K", param_values=param_values)

plt.subplot(1, 2, 1)
plt.plot(param_values, test_accuracies)
plt.xlabel('K')
plt.ylabel("Test accuracy(K)")

data_pert_gaussian = CDataPerturbGaussian()
xp2 = data_pert_gaussian.perturb_dataset(x)
test_accuracies = robustness_test(clf, data_pert_gaussian, param_name="sigma", param_values=param_values)

plt.subplot(1, 2, 2)
plt.plot(param_values, test_accuracies)
plt.xlabel('Sigma')
plt.ylabel("Test accuracy(Sigma)")

plt.show()

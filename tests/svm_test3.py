
import numpy as np

from SVM.GaussianSVM import svm as gaussian_svm

def _read(filename):
    result = np.genfromtxt(filename, delimiter=',')
    _r, _c = result.shape
    np.random.shuffle(result)
    return result[:, :_c-1], result[:, _c-1]


def _process_boolean_labels(Y):
    return np.where(Y == 0, -1, Y)

def main():
    # get and process data and labels
    X, Y = _read('data/hw4/bank-note/train.csv')
    Y = _process_boolean_labels(Y)

    C_nominator_values = [100, 500, 700]
    gamma_values = [0.1, 0.5, 1, 5, 100]

    for C_nominator in C_nominator_values:
        C = C_nominator / (X.shape[0] + 1)
        print("C: {}/{}".format(C_nominator, X.shape[0] + 1))
        for gamma in gamma_values:
            print("gamma: {}".format(gamma))
            # create svm object
            clf_gaussian = gaussian_svm(gamma, C, debug=True)
            # train svm
            w_f = clf_gaussian.fit(X, Y)
            print("Final weight vector: {}".format(w_f))

            # predict
            X_test, Y_test = _read('data/hw4/bank-note/test.csv')
            Y_test = _process_boolean_labels(Y_test)
            Y_hat = clf_gaussian.predict(X_test)

            # evaluate accuracy
            accuracy = np.sum(Y_hat == Y_test) / Y_test.shape[0]
            print("accuracy: {}".format(accuracy))



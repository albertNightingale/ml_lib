import numpy as np

from SVM.DualSVM import svm as dual_svm

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
    
    for C_nominator in C_nominator_values:
        C = C_nominator / (X.shape[0] + 1)
        print("C: {}/{}".format(C_nominator, X.shape[0] + 1))

        # create svm object
        primal_cfr = dual_svm(C, debug=True)
        # train svm
        w_f, b_f = primal_cfr.fit(X, Y)
        print("Final weight vector: {}".format(w_f))
        print("Final bias: {}".format(b_f))

        # predict training data and evaluate accuracy
        Y_hat_train = primal_cfr.predict(X)
        accuracy_train = np.sum(Y_hat_train == Y) / Y.shape[0]
        print("accuracy on training data: {}".format(accuracy_train))

        # predict test data and evaluate accuracy
        X_test, Y_test = _read('data/hw4/bank-note/test.csv')
        Y_test = _process_boolean_labels(Y_test)
        Y_hat_test = primal_cfr.predict(X_test)
        accuracy_test = np.sum(Y_hat_test == Y_test) / Y_test.shape[0]
        print("accuracy on testing data: {}".format(accuracy_test))

        print()



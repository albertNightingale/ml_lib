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

    C_values = [100/873, 500/873, 700/873]
    
    # create svm object
    primal_cfr = dual_svm(C_values[1], debug=True)
    # train svm
    w_f, b_f = primal_cfr.fit(X, Y)
    print("Final weight vector: {}".format(w_f))
    print("Final bias: {}".format(b_f))

    # predict
    X_test, Y_test = _read('data/hw4/bank-note/test.csv')
    Y_test = _process_boolean_labels(Y_test)
    Y_hat = primal_cfr.predict(X_test)

    # evaluate accuracy
    accuracy = np.sum(Y_hat == Y_test) / Y_test.shape[0]
    print("accuracy: {}".format(accuracy))



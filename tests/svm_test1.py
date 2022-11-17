import numpy as np

from SVM.PrimalSVM import svm as std_svm

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
    cfr_std = std_svm(10, C_values[1], 100, lr_scheduler="b", debug=True)
    # train svm
    w_f = cfr_std.fit(X, Y)
    print("Final weight vector: {}".format(w_f))

    # predict
    X_test, Y_test = _read('data/hw4/bank-note/test.csv')
    Y_test = _process_boolean_labels(Y_test)
    Y_hat = cfr_std.predict(X_test)

    # evaluate accuracy
    accuracy = np.sum(Y_hat == Y_test) / Y_test.shape[0]
    print("accuracy: {}".format(accuracy))



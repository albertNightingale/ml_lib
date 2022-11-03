import numpy as np

from Perceptron.StandardPerceptron import perceptron


def _read(filename):
    result = np.genfromtxt(filename, delimiter=',')
    _r, _c = result.shape
    return result[:, :_c-1], result[:, _c-1]

def _process_boolean_labels(Y):
    return np.where(Y == 0, -1, Y)

def main():
    # get and process data and labels
    X, Y = _read('data/hw3/bank-note/train.csv')
    Y = _process_boolean_labels(Y)
    
    # create perceptron object
    cfr = perceptron(10)
    w_f = cfr.fit(X, Y)
    print("Final weight vector: {}".format(w_f))

    # predict
    X_test, Y_test = _read('data/hw3/bank-note/test.csv')
    Y_test = _process_boolean_labels(Y_test)
    Y_hat = cfr.predict(X_test)
    # print("predicted result: ")
    # print(Y_hat)

    # evaluate accuracy
    accuracy = np.sum(Y_hat == Y_test) / Y_test.shape[0]
    print("accuracy: {}".format(accuracy))
import numpy as np

from Perceptron.AveragePerceptron import perceptron as avg_perceptron
from Perceptron.VotedPerceptron import perceptron as vote_perceptron


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
    cfr_avg = avg_perceptron(10)
    weight_sum = cfr_avg.fit(X, Y)
    print("Sum weight vector: {}".format(weight_sum))

    # predict
    X_test, Y_test = _read('data/hw3/bank-note/test.csv')
    Y_test = _process_boolean_labels(Y_test)
    Y_hat = cfr_avg.predict(X_test)
    # print("predicted result: ")
    # print(Y_hat)

    # evaluate accuracy
    accuracy = np.sum(Y_hat == Y_test) / Y_test.shape[0]
    print("accuracy: {}".format(accuracy))

    # create vote perceptron classifier and compare weights with average perceptron
    # create perceptron object
    cfr_vote = vote_perceptron(10)
    weights, counts = cfr_vote.fit(X, Y)
    sum_by_col = np.zeros(weights.shape[1])
    print("vote perceptron")
    for j in range(weights.shape[1]):
        sum_by_col[j] = weights[:,j].dot(counts)
    print("sum by column:", sum_by_col)

"""
HW 2 problem 4a
"""
import numpy as np
import copy

from LinearRegression.LinearRegression import gradient_descent

train_file = "data/hw2/concrete/train.csv"
test_file = "data/hw2/concrete/test.csv"

def _read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data

def process(file):
    data = _read(file)
    num_cols = len(data[0].strip().split(','))
    S = np.empty((len(data), num_cols), dtype=object)
    for i in range(len(data)):
        S[i] = np.array(data[i].strip().split(',')).astype(float)
    return S


def main(): 
    train_data = copy.deepcopy(process(train_file)) 
    test_data = copy.deepcopy(process(test_file)) 
    shape = train_data.shape

    learning_rate = 1
    w = np.full(shape[1]-1, 0)

    weight = gradient_descent(train_data, learning_rate, w, type="batch")
    print()
    print(weight)
    for test in test_data:
        data = test[:-1]
        label = test[-1]
        print("data: {}, label: {}, prediction: {}".format(data, label, np.dot(weight, data)))
        expected_label = np.dot(weight, data)
        if label is not expected_label:
            print("Expected: {}, Actual: {}".format(label, expected_label))

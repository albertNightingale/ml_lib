import numpy as np
import math

from LinearRegression.util import lms_cost, lms_batch_gradient

def gradient_descent(training_data, learning_rate, initial_w=None, type="batch"):
    w = initial_w
    if initial_w is None:
        w = np.zeros(training_data.shape[1])
    
    if type == "batch":
        w = _batch_gradient_descent(training_data, learning_rate, w)
    
    return w


def _batch_gradient_descent(training_data, r, w):
    X = training_data[:, :-1]
    Y = training_data[:, -1]
    diff = 1
    iteration = 0

    while np.linalg.norm(diff) > 0.000001:
        # update weight
        diff = r * lms_batch_gradient(X, Y, w)
        w = w - diff
        # update learning rate
        r = r * 0.5
        print("iteration: {}, r: {}, w: {}, cost: {}".format(iteration, r, w, lms_cost(X, Y, w)))
        iteration += 1
    return w


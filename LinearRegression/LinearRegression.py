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


def _batch_gradient_descent(training_data, learning_rate, w):
    X = training_data[:, :-1]
    Y = training_data[:, -1]
    J = math.inf
    
    while J > 0.01:
        w = w - learning_rate * lms_batch_gradient(X, Y, w)
    return w


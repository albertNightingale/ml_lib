
import numpy as np

def lms_cost(X, Y, weight):
    J_sum = 0
    for i in range(X.shape[0]):
        J_sum += (Y[i] - weight.T * X[i]) ** 2
    return 0.5 * J_sum


def lms_batch_gradient(X, Y, w):
    shape = X.shape
    sum = np.zeros(shape[1])
    for j in range(shape[1]):
        for i in range(shape[0]):
            sum[j] += - (Y[i] - w.T * X[i]) * X[i][j]
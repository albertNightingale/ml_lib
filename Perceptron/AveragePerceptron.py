import numpy as np

class perceptron:
    def __init__(self, epoch, debug=False):
        self.epoch = epoch
        self.debug = debug
        self.weight_sum = None

    def fit(self, X, Y):
        r = 1
        w_t = np.zeros(X.shape[1])

        w_sum = np.zeros(X.shape[1])
        for T in range(self.epoch):
            self.debug and print("Epoch: {}, weight: {}".format(T, w_t))
            for x_i, y_i in zip(X, Y):
                y_hat = w_t.T.dot(x_i)
                if y_hat * y_i <= 0: # y != y_hat
                    w_t = w_t + r * y_i * x_i
                w_sum += w_t
        # self.a_weight = w_sum / (self.epoch * X.shape[0])
        self.weight_sum = w_sum
        return self.weight_sum


    def predict(self, X):
        label_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            label_hat[i] = np.sign(self.weight_sum.T.dot(X[i]))
        return label_hat


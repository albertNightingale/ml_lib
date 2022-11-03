import numpy as np

class perceptron:
    def __init__(self, epoch, debug=False):
        self.epoch = epoch
        self.debug = debug
        self.w_f = None

    def fit(self, X, Y):
        r = 1
        w_t = np.zeros(X.shape[1])
        for T in range(self.epoch):
            self.debug and print("Epoch: {}, weight: {}".format(T, w_t))
            for x_i, y_i in zip(X, Y):
                y_hat = w_t.T.dot(x_i)
                if y_hat * y_i <= 0: # y != y_hat
                    w_t = w_t + r * y_i * x_i
        self.w_f = w_t
        return w_t


    def predict(self, X):
        label_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            label_hat[i] = np.sign(self.w_f.T.dot(X[i]))
        return label_hat


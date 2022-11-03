import numpy as np

class perceptron:
    def __init__(self, epoch, debug=False):
        self.epoch = epoch
        self.debug = debug
        self.weights = None
        self.votes = None

    def fit(self, X, Y):
        r = 1
        w_t = np.zeros(X.shape[1])
        c_t = 1

        used_w = []
        used_c = []
        for T in range(self.epoch):
            self.debug and print("Epoch: {}, used_weight size: {}".format(T, len(used_w)))
            for x_i, y_i in zip(X, Y):
                y_hat = w_t.T.dot(x_i)
                if y_hat * y_i <= 0: # y != y_hat
                    used_w.append(w_t)
                    used_c.append(c_t)
                    w_t = w_t + r * y_i * x_i
                    c_t = 1
                else:
                    c_t += 1

        self.weights = np.array(used_w)
        self.votes = np.array(used_c)
        return self.weights, self.votes


    def predict(self, X):
        label_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            x = X[i]
            for w_i, v_i in zip(self.weights, self.votes):
                label_hat[i] += np.sign(w_i.T.dot(x)) * v_i
        return np.sign(label_hat)


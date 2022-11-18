import numpy as np

class svm:
    def __init__(self, epoch, C, N, lr_scheduler="a", debug=False):
        # hyperparameters
        self.C = C
        self.N = N
        self.epoch = epoch
        
        self.lr_scheduler = lr_scheduler
        self.gamma_0 = 0.2
        self.a = 0.3 if self.lr_scheduler == "a" else None

        # settings
        self.debug = debug
        
        # settings for fitting data
        self.w_0 = None
        self.weight_final = None
        
        self.debug and print("SVM: epoch={}, C={}, N={}, lr_scheduler={} gamma_0={} a={}".format(self.epoch, self.C, self.N, self.lr_scheduler, self.gamma_0, self.a))

    def a_schedule_gamma(self, T):
        return self.gamma_0 / (1 + self.gamma_0 * T / self.a)

    def b_schedule_gamma(self, T):
        return self.gamma_0 / (1 + T)

    def fit(self, X, Y):
        self.w_0 = np.zeros(X.shape[1]+1) # augmented w_0
        w = self.w_0
        
        gamma_t = self.gamma_0
        lr_schedule = self.a_schedule_gamma if self.lr_scheduler == "a" else self.b_schedule_gamma

        for T in range(self.epoch):
            merged_data = np.column_stack((X, Y))
            np.random.shuffle(merged_data)
            X, Y = merged_data[:, :-1], merged_data[:, -1]

            for x_i, y_i in zip(X, Y):
                # randomly select ONE data point
                # index = np.random.choice(np.arange(X.shape[0]), 1)
                # y_i = Y[index] 
                x_i = np.append(x_i, 1) # augmented x_i
                if y_i * w.T.dot(x_i) <= 1: # expected vs actual is not correct
                    gradient = gamma_t * (self.w_0 - self.N * self.C * y_i * x_i)
                    w -= gradient
                else:
                    self.w_0 = (1 - gamma_t) * self.w_0
                # update gamma_t
                gamma_t = lr_schedule(T) 

        self.weight_final = w
        return w
    
    def predict(self, X):
        label_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            augmented_x_i = np.append(X[i], 1)
            _prod = self.weight_final.T.dot(augmented_x_i)
            label_hat[i] = _prod
            
        return np.sign(label_hat)
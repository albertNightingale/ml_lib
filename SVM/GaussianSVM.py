import numpy as np
from scipy.optimize import minimize as min

class svm:
    def __init__(self, gamma, C, debug=False):
        # hyperparameters
        self.C = C
        self.gamma = gamma

        # settings
        self.debug = debug
        
        # settings for fitting data
        self.train_X = None
        self.train_Y = None
        self.alpha_final = None
        self.weight_final = None
        self.bias_final = None
        
        self.debug and print("Initialized SVM with C = {}".format(self.C))

    def fit(self, X, Y):
        self.train_X = X
        self.train_Y = Y
        self.alpha_final = None

        self.weight_final = np.zeros(X.shape[1], )
        self.bias_final = 0

        alpha = np.full(X.shape[0], self.C/2)
        cons = [{'type': 'eq', 'fun': svm.constraint, 'jac': svm.constraint_jac, 'args': [Y]}]
        bnds = [(0, self.C) for i in range(X.shape[0])]
        result_alpha = min(fun=svm.dual_svm_objective, jac=svm.jac, x0=alpha, args=[X, Y], bounds=bnds, method='SLSQP', constraints=cons)
        print("success {}".format(result_alpha.success))
        print("message {}".format(result_alpha.message))
        print("alpha {}".format(result_alpha.x))
        self.alpha_final = result_alpha.x
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                self.weight_final[i] += self.alpha_final[j] * Y[j] * X[j][i]
        
        self.bias_final = np.mean(Y - np.dot(self.weight_final, X.T))

        return self.weight_final, self.bias_final

    # takes gradient against alpha
    def jac(alpha, arguments):
        X, Y = arguments[0], arguments[1]
        inner_prod = np.multiply(np.matmul(X, X.T), np.matmul(Y, Y.T))
        return np.subtract(np.matmul(inner_prod, alpha), np.ones(alpha.shape[0]))
    
    def dual_svm_objective(alpha, arguments):
        X, Y = arguments[0], arguments[1]
        # print("dual objective")
        # print(X.shape)
        # print(Y.shape)

        inner_prod = np.multiply(np.matmul(X, X.T), np.matmul(Y, Y.T))
        outer_prod = np.matmul(np.matmul(alpha.T, inner_prod), alpha)
        return 0.5 * outer_prod - np.sum(alpha)
    
    # sum of alpha_i * y_i = 0
    def constraint(alpha, y):
        return np.dot(alpha, y)

    def constraint_jac(alpha, y):
        return y

    def K(self, x1, x2):
        return np.exp(- 1 / self.gamma * np.linalg.norm(x1 - x2)**2)

    def predict(self, X):
        
        y_hat = np.zeros(X.shape[0], )
        for i in range(X.shape[0]):
            for j in range(self.train_X.shape[0]):
                _train_x = self.train_X[j]
                _train_y = self.train_Y[j]
                _alpha = self.alpha_final[j]
                y_hat[i] += _alpha * _train_y * self.K(_train_x, X[i])
            
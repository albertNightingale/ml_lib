import numpy as np
from scipy.optimize import minimize as min

class svm:
    def __init__(self, C, debug=False):
        # hyperparameters
        self.C = C

        # settings
        self.debug = debug
        
        # settings for fitting data
        self.weight_final = None
        self.bias_final = None
        
        self.debug and print("Initialized SVM with C = {}".format(self.C))

    def fit(self, X, Y):
        alpha = np.zeros(X.shape[0])
        cons = [
            {'type': 'eq', 'fun': svm.constraint_1}, 
            {'type': 'eq', 'fun': svm.constraint_2, 'args': [self.C]}, 
            {'type': 'eq', 'fun': svm.constraint_3, 'args': [Y]}]
        result_alpha = min(fun=svm.dual_svm_objective, x0=alpha, args=[alpha, X, Y], constraints=cons)
        print("result_alpha {}".format(result_alpha))
        self.weight_final = np.dot(np.dot(result_alpha, Y), X)
        self.bias_final = np.mean(Y - np.dot(self.weight_final, X.T))

        return self.weight_final, self.bias_final
    
    def dual_svm_objective(alpha, arguments):
        X, Y = arguments[0], arguments[1]
        # print("dual objective")
        # print(X.shape)
        # print(Y.shape)

        inner_prod = np.multiply(np.matmul(X, X.T), np.matmul(Y, Y.T))
        outer_prod = np.matmul(np.matmul(alpha.T, inner_prod), alpha)
        return 0.5 * outer_prod - np.sum(alpha)

    # alpha_i > 0
    def constraint_1(alpha):
        for a_i in alpha:
            if a_i < 0:
                return -1
        return 1
    
    # alpha_i < C
    def constraint_2(alpha, C):
        for a_i in alpha:
            if a_i > C:
                return -1
        return 1
    
    # sum of alpha_i * y_i = 0
    def constraint_3(alpha, y):
        return np.dot(alpha, y)


    def predict(self, X):
        label_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            augmented_x_i = np.append(X[i], 1)
            _prod = self.weight_final.T.dot(augmented_x_i) + self.bias_final
            label_hat[i] = _prod
            
        return np.sign(label_hat)
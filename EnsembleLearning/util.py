import numpy as np
import math

def get_alpha(epsilon):
    return 0.5 * math.log((1 - epsilon) / epsilon)

def get_distribution(old_distribution, alpha, incorrect_indices):
    distribution_size = len(old_distribution)
    
    new_distribution = np.empty(distribution_size)
    for i in range(len(new_distribution)):
        if i in incorrect_indices:
            new_distribution[i] = old_distribution[i] * math.exp(alpha)
        else:
            new_distribution[i] = old_distribution[i] * math.exp(-alpha)    

    # multiply by 1/Z_t, where Z_t is the normalization factor
    weight = new_distribution / np.sum(new_distribution)
    return weight
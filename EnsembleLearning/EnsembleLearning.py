import numpy as np

from DecisionTree.ID3 import ID3, assess_id3, traverse_one

from EnsembleLearning.util import get_alpha
from EnsembleLearning.util import get_distribution

def adaboost(S, attr_dict, attr_col_map, T = 500):

    # initialize weights 
    weight = np.full(len(S), 1/len(S))
    # initialize classifiers
    classifiers = np.empty(T, dtype=object)
    alpha = np.zeros(T, dtype=float)

    for i in range(T):
        # Train a weak learner
        classifiers[i] = ID3(S, attr_dict, attr_col_map, maximum_depth=1, IG_algotithm="entropy", weight=weight)
        # test the weak learner
        error_rate, incorrect_indices = assess_id3(classifiers[i], S, attr_col_map, attr_dict)
        # Update the alpha
        alpha[i] = get_alpha(error_rate)
        # Update the weights
        weight = get_distribution(weight, alpha[i], incorrect_indices)

    return classifiers, alpha


def assess_adaboost(classifiers, alpha, dataset, attr_col_map, attr_map, label_values: list):
    incorrect_count = 0
    incorrect_indices = []
    total = len(dataset)

    for data in dataset:
        predicted_label = get_H_final(data, classifiers, alpha, attr_col_map, attr_map, label_values)
        actual_label = data[-2] # label value
        if predicted_label != actual_label:
            incorrect_count += 1
            incorrect_index = data[-1]
            incorrect_indices.append(incorrect_index)
    return incorrect_count/total, incorrect_indices


def get_H_final(data, classifiers, alpha, attr_col_map, attr_map, label_values):
    H_final = 0
    for i in range(len(classifiers)):
        # traverse the tree and get the label
        actual_label_i = traverse_one(classifiers[i], data, attr_col_map, attr_map)
        # get the index of the label
        label_index_i = label_values.index(actual_label_i)
        # get the alpha
        alpha_i = alpha[i]
        # update the expected label
        H_final += alpha_i * label_index_i
    
    # get the index of the label
    label_index = int(H_final % len(label_values))
    return label_values[label_index]
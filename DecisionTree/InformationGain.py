import numpy as np
import math

from DecisionTree.helper import getValueAndFrequency
from DecisionTree.helper import getColumnAndLabel

"""
counts: np array of occurance of a label in S, it is size of number of labels. 
total: size of S
"""


def entropy(counts, total):
    ratios = np.divide(counts, total)
    e = 0
    for ratio in ratios:
        e += ratio * math.log(ratio, 2)
    return -e


'''
given S and attributes map
returns index in the attribute with the best IG
'''


def IG(S, attributes):
    attribute_len = len(attributes)

    # entropy of S
    S_size = len(S)
    _labels, counts = getValueAndFrequency(S, 7)
    H_S = entropy(counts, S_size)

    # IG of all columns
    all_columns_IG = np.full(attribute_len, H_S)

    for attribute_index in range(attribute_len):
        # calculating IG of one attribute
        # attribute_label is certain attribute and label columns from S
        attribute_label = getColumnAndLabel(S, attribute_index)
        for attribute_value in attributes[attribute_index]:
            attribute_of_value = attribute_label[np.where(
                attribute_label[:, 0] == attribute_value)]
            _attributes, counts = getValueAndFrequency(attribute_of_value, 1)
            _entropy = entropy(counts, len(attribute_of_value))
            # subtract that attribute value entropy
            all_columns_IG[attribute_index] -= len(
                attribute_of_value)/S_size * _entropy

            print("attribute value:", attribute_value)
            print("values mapping")
            print("values", _attributes)
            print("counts", counts)
            print("entropy for \'" + str(attribute_value) + "\' is ", _entropy)
            print()

    print("IG of all columns:", all_columns_IG)
    return np.argmax(all_columns_IG)

import numpy as np
import math

import DecisionTree.util as util


IG_debug = False

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
returns name in the attribute with the best IG
'''
def IG(S, remaining_attributes, attr_col_mapping):
    attribute_len = len(attr_col_mapping)
    
    # entropy of S
    S_size = len(S)
    _labels, counts = util.getValueAndFrequency(S, 6)
    H_S = entropy(counts, S_size)

    # IG of columns
    all_columns_IG = {}
    for attribute_name in remaining_attributes:
        all_columns_IG[attribute_name] = H_S

    # calculating IG of each attributes in remaining_attributes
    for attribute_name in remaining_attributes:
        attribute_label = util.getColumnAndLabel(S, attr_col_mapping[attribute_name])  # attribute_label is certain attribute and label columns from S
        for attribute_value in remaining_attributes[attribute_name]:
            data_with_attribute_value = attribute_label[np.where(attribute_label[:,0]==attribute_value)[0]] 
            _attributes, counts = util.getValueAndFrequency(data_with_attribute_value, 1)
            _entropy = entropy(counts, len(data_with_attribute_value))
            # subtract that attribute value entropy
            all_columns_IG[attribute_name] -= len(data_with_attribute_value)/S_size * _entropy

            IG_debug and print("attribute value:", attribute_value)
            IG_debug and print("values mapping")
            IG_debug and print("values", _attributes)
            IG_debug and print("counts", counts)
            IG_debug and print("entropy for \'" + str(attribute_value) + "\' is ", _entropy) 
            IG_debug and print()

    IG_debug and print("IG of all columns:", all_columns_IG)
    
    # return attribute name with highest IG
    curr_max_attr = None
    curr_max_IG = 0
    
    for attribute_name in all_columns_IG:
        if all_columns_IG[attribute_name] >= curr_max_IG:
            curr_max_attr = attribute_name
            curr_max_IG = all_columns_IG[attribute_name]
    IG_debug and print("Highest of all:", curr_max_attr, ", IG score:", curr_max_IG)
    return curr_max_attr

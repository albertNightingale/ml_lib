import numpy as np
import math

import DecisionTree.Config as config
import DecisionTree.util as util


IG_debug = False

"""
counts: np array of occurance of a label in S, it is size of number of labels. 
total: size of S
"""
def entropy(counts, total):
    if len(counts) == 0 or total == 0:
        return 0

    ratios = np.divide(counts, total)
    e = 0
    for ratio in ratios:
        e += ratio * math.log(ratio, 2)
    return -e

def gini_index(counts, total):
    if len(counts) == 0 or total == 0:
        return 0

    ratios = np.divide(counts, total)
    power = np.power(ratios, 2)
    return 1 - np.sum(power)

def majority_error(counts, total):
    if len(counts) == 0:
        return 0
    max_count = counts[np.argmax(counts)]
    return (total - max_count) / max_count

'''
given S and attributes map
returns name in the attribute with the best IG
'''
def IG(S, remaining_attributes, attr_col_mapping, cfg: config):
    attribute_len = len(attr_col_mapping)
    IG_method = entropy
    if cfg.get_IG_algotithm() == "gini_index":
        IG_method = gini_index
    elif cfg.get_IG_algotithm() == "majority_error":
        IG_method = majority_error
    else: 
        IG_method = entropy
    
    # entropy of S
    S_size = len(S)
    _labels, counts = util.getValueAndFrequency(S, cfg.get_label_column())

    _main_result = IG_method(counts, S_size)

    # intialize IG of all available attributes
    all_columns_IG = {}
    for attribute_name in remaining_attributes:
        all_columns_IG[attribute_name] = _main_result

    # calculating IG of each attributes in remaining_attributes
    for attribute_name in remaining_attributes:
        attribute_label = util.getColumnAndLabel(S, attr_col_mapping[attribute_name], cfg)  # attribute_label is certain attribute and label columns from S
        attribute = remaining_attributes[attribute_name]
        
        for attribute_value in attribute.get_values():
            S_v = attribute_label[attribute_label[:,0]==attribute_value] 
            _labels, _counts = util.getValueAndFrequency(S_v, 1)
            _result = IG_method(counts, len(S_v))
                
            # subtract the result from _main_result
            all_columns_IG[attribute_name] -= len(S_v)/S_size * _result

            IG_debug and print("attribute value:", attribute_value)
            IG_debug and print("values mapping")
            IG_debug and print("values", _labels)
            IG_debug and print("counts", _counts)
            IG_debug and print("entropy for \'" + str(attribute_value) + "\' is ", _result) 
            IG_debug and print()

    IG_debug and print("IG of all columns:", all_columns_IG)
    
    # return attribute name with highest IG
    curr_max_attr = None
    curr_max_IG = None
    
    for attribute_name in all_columns_IG:
        if curr_max_IG == None or all_columns_IG[attribute_name] >= curr_max_IG:
            curr_max_attr = attribute_name
            curr_max_IG = all_columns_IG[attribute_name]
    IG_debug and print("Highest of all:", curr_max_attr, ", IG score:", curr_max_IG)
    return curr_max_attr

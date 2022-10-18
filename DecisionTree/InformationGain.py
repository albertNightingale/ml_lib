import numpy as np
import math

import DecisionTree.Config as config
import DecisionTree.util as util


IG_debug = False

"""
counts: np array of occurance of a label in S, it is size of number of labels. 
total: size of S
"""
def entropy(total=None, partitions_idx=None, weight=None, normal_weight=None):

    if total == 0:
        return 0
    
    e = 0
    if weight is None:
        for partition in partitions_idx:
            p = len(partition) / total
            e += p * math.log(p, 2)
    else:
        for partition in partitions_idx:
            partitioned_weight = np.take(weight, partition.astype(int))
            p = np.sum(np.divide(partitioned_weight, normal_weight)) / total
            e += p * math.log(p, 2)
    return -e
    

def gini_index(total=None, partitions_idx=None, weight=None, normal_weight=None):
    if total == 0:
        return 0
    
    gini = 1
    if weight is None:
        for partition in partitions_idx:
            p = len(partition) / total
            gini -= p * p
    else:
        for partition in partitions_idx:
            p = np.sum(np.divide(weight[partition], normal_weight)) / total
            gini -= p * p

    return gini

def majority_error(total=None, partitions_idx=None, weight=None, normal_weight=None):

    counts = np.empty(len(partitions_idx))

    if weight is not None:
        for i in range(len(partitions_idx)):
            partition = partitions_idx[i]
            counts[i] = np.sum(np.divide(weight[partition], normal_weight))
    else:
        for i in range(len(partitions_idx)):
            partition = partitions_idx[i]
            counts[i] = len(partition)

    if len(counts) == 0:
        return 0
        
    return (total - counts[np.argmax(counts)]) / total

'''
given S and attributes map
returns name in the attribute with the best IG
'''
def IG(S, remaining_attributes, attr_col_mapping, cfg: config):
    # determine the IG method to use
    IG_method = entropy
    if cfg.get_IG_algotithm() == "gini_index":
        IG_method = gini_index
    elif cfg.get_IG_algotithm() == "majority_error":
        IG_method = majority_error
    else: 
        IG_method = entropy
    
    # entropy of S
    total_size = len(S)
    labels, counts, partitions_idx = util.getAttributeFrequency(S, cfg.get_label_column(), cfg.get_index_column())

    _main_result = IG_method(total_size, partitions_idx, weight=cfg.get_weight(), normal_weight=cfg.get_normal_weight())

    # intialize IG of all available attributes
    all_columns_IG = {}
    for attribute_name in remaining_attributes:
        all_columns_IG[attribute_name] = _main_result 

    # calculating IG of each attributes in remaining_attributes
    for attribute_name in remaining_attributes:
        attribute_index = attr_col_mapping[attribute_name]
        attribute = remaining_attributes[attribute_name] 
        
        for av in attribute.get_values():
            S_av = S[S[:,attribute_index]==av] # subset where the attribute value of each data equals to av 
            _labels, _counts, _partitions_idx = util.getAttributeFrequency(S_av, cfg.get_label_column(), cfg.get_index_column())
            _result = IG_method(len(S_av), _partitions_idx, weight=cfg.get_weight(), normal_weight=cfg.get_normal_weight())
            
            # subtract the result from _main_result
            all_columns_IG[attribute_name] -= len(S_av)/total_size * _result

            IG_debug and print("attribute value:", av)
            IG_debug and print("values mapping")
            IG_debug and print("values", _labels)
            IG_debug and print("counts", _counts)
            IG_debug and print("entropy for \'" + str(av) + "\' is ", _result) 
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

import numpy as np
import copy

import DecisionTree.util as util

# compute the median of the attribute column of the dataset where that attribute is numeric
# update the values of attribute object with the median as dividing line
def convertNumericToBinary(dataset, attr_col_map, attributes):
    _attr_copy = copy.deepcopy(attributes)
    
    for attr in attributes:
        if attributes[attr].get_type() == "numeric":
            col_index = attr_col_map[attr]
            median = np.median(dataset[:, col_index].astype(np.float))
            _attr_copy[attr].set_median(median)
            _attr_copy[attr].set_values([False, True])
            
            # set the value of the attribute in data to be binary
            for data in dataset:
                if float(data[col_index]) <= median:
                    data[col_index] = _attr_copy[attr].get_values()[0]
                else:
                    data[col_index] = _attr_copy[attr].get_values()[1]
    return _attr_copy, dataset

def normalizeBinary(dataset, attr_col_map, attributes):
    for attr in attributes:
        if attributes[attr].get_type() == "binary":
            col_index = attr_col_map[attr]
            for data in dataset:
                if data[col_index] == "Yes":
                    data[col_index] = True
                else:
                    data[col_index] = False
    return attributes, dataset

"""
normalizes attribute values that are equal to unknown in dataset to the most common value of the same attribute in the dataset
"""
def normalizeUnknownAttributeValue(dataset, attr_col_map, attributes):
    for attr_name in attributes:
        attr = attributes[attr_name]
        if attr.get_type() == "categorical" and "unknown" in attr.get_values():
            col_index = attr_col_map[attr_name] # get the column index of the attribute

            # get the most common value of the attribute
            _value, _counts = util.getValueAndFrequency(dataset, col_index)
            most_common_value = _value[np.argmax(_counts)]

            for data in dataset:
                if data[col_index] == "unknown":
                    data[col_index] = most_common_value
    
    return dataset
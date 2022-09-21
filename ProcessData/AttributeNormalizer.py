import numpy as np
import copy

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
                    data[col_index] = False
                else:
                    data[col_index] = True
    return _attr_copy, dataset

def normalizeBinary(dataset, attr_col_map, attributes):
    for attr in attributes:
        if attributes[attr].get_type() == "binary":
            col_index = attr_col_map[attr]
            for data in dataset:
                if data[col_index] == "Yes":
                    data[col_index] = False
                else:
                    data[col_index] = True
    return attributes, dataset
from array import array
from DecisionTree.Config import config
import numpy as np

"""find the most common label in S, returns the value of the most common label"""
def findMostCommonLabel(S, cfg):
    value, count, partitions_by_value = getAttributeFrequency(S, cfg.get_label_column())
    return value[count.argmax()]

"""get value and frequency and partitioned index of S in a specific column"""
def getAttributeFrequency(S, column_index, index_column_idx=None):
    values, counts = np.unique(S[:, column_index], return_counts=True)
    partitions_by_value = np.empty(len(values), dtype=array)

    if index_column_idx == None:
        for i in range(len(values)): 
            v = values[i]
            partitions_by_value[i] = np.argwhere(S[:, column_index] == v).flatten()
    else:
        for i in range(len(values)): 
            v = values[i]
            partitions_by_value[i] = S[S[:, column_index] == v][:, index_column_idx]

    return values, counts, partitions_by_value

"""get a column at index and label in a 2 column mappings"""
def getColumnAndLabel(S, index, cfg):
    return np.column_stack((S[:, index], S[:, cfg.get_label_column()]))

"""get the row index column of S"""
def getRowIndexColumn(S):
    return S[:, S.shape[1] - 1]
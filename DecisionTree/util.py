import numpy as np

"""find the most common label in S, returns the value of the most common label"""
def findMostCommonLabel(S, cfg):
    value, count = getAttributeFrequency(S, cfg.get_label_column())
    return value[count.argmax()]

"""get value and frequency of a specific column"""
def getAttributeFrequency(S, column_index):
    return np.unique(S[:, column_index], return_counts=True)

"""get a column at index and label in a 2 column mappings"""
def getColumnAndLabel(S, index, cfg):
    return np.column_stack((S[:, index], S[:, cfg.get_label_column()]))

"""get the row index column of S"""
def getRowIndexColumn(S):
    return S[:, S.shape[1] - 1]
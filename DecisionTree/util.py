import numpy as np

"""find the most common label in S, returns the value of the most common label"""
def findMostCommonLabel(S):
    value, count = getValueAndFrequency(S, 6)
    return value[count.argmax()]

"""get value and frequency of a specific column"""
def getValueAndFrequency(S, column_index):
    return np.unique(S[:, column_index], return_counts=True)

"""get a column at index and label in a 2 column mappings"""
def getColumnAndLabel(S, index):
    return np.column_stack((S[:, index], S[:, 6]))
import numpy as np

from DecisionTree.ID3 import ID3
from DecisionTree.ID3 import traverse
from DecisionTree.Config import config

attributes = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"]
}

attr_col_map = {
    "buying": 0,
    "maint": 1,
    "doors": 2,
    "persons": 3,
    "lug_boot": 4,
    "safety": 5,
}

labels = ["unacc", "acc", "good", "vgood"]

train_file = "data/hw1/car/train.csv"

def _read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data

def process(file):
    data = _read(file)
    S = np.empty((len(data), 7), dtype=object)
    for i in range(len(data)):
        S[i] = data[i].strip().split(',')
    return S

def __main__(): 
    data = process(train_file)
    tree = ID3(data, attributes, attr_col_map, maximum_depth=6)
    correct_ratio, incorrect_ratio = traverse(tree, data, attr_col_map)
    print("correct_ratio:", correct_ratio)
    print("incorrect_ratio:", incorrect_ratio)

__main__()
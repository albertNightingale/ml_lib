"""
HW 1 problem 2a, b, c
"""
import numpy as np

from DecisionTree.ID3 import ID3
from DecisionTree.ID3 import traverse

from ProcessData.Attribute import Attribute

attributes = {
    "buying": Attribute("buying", "categorical", ["vhigh", "high", "med", "low"]),
    "maint": Attribute("maint", "categorical", ["vhigh", "high", "med", "low"]),
    "doors": Attribute("doors", "categorical", ["2", "3", "4", "5more"]),
    "persons": Attribute("persons", "categorical", ["2", "4", "more"]),
    "lug_boot": Attribute("lug_boot", "categorical", ["small", "med", "big"]),
    "safety": Attribute("safety", "categorical", ["low", "med", "high"])
}

# attribute to column index map
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
test_file = "data/hw1/car/test.csv"

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

def main(): 
    data = process(train_file)
    test_data = process(test_file)

    methods2test = ["entropy", "gini_index", "majority_error"]
    depth2test = [1, 2, 3, 4, 5, 6]

    print("testing with training data")
    for depth in depth2test:
        print("*****depth:", depth, "*****")
        for method in methods2test:
            print("---| method:", method)
            tree = ID3(data, attributes, attr_col_map, maximum_depth=depth, IG_algotithm=method)
            correct_ratio, incorrect_ratio = traverse(tree, data, attr_col_map, attributes)
            print("------| correct_ratio:", correct_ratio)
    
    print("testing with the test data")
    for depth in depth2test:
        print("*****depth:", depth, "*****")
        for method in methods2test:
            print("---| method:", method)
            tree = ID3(data, attributes, attr_col_map, maximum_depth=depth, IG_algotithm=method)
            correct_ratio, incorrect_ratio = traverse(tree, test_data, attr_col_map, attributes)
            print("------| correct_ratio:", correct_ratio)

if __name__ == "__main__":
    main()
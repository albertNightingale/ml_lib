import numpy as np
from DecisionTree.ID3 import ID3
from DecisionTree.ID3 import traverse

from ProcessData.Attribute import Attribute
from ProcessData.AttributeNormalizer import convertNumericToBinary
from ProcessData.AttributeNormalizer import normalizeBinary

attributes = {
    "age": Attribute("age", "numeric", None),
    "job": Attribute("job", "categorical", ["admin", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]),
    "marital": Attribute("marital", "categorical", ["divorced", "married", "single"]),
    "education": Attribute("education", "categorical", ["primary", "secondary", "tertiary", "unknown"]),
    "default": Attribute("default", "binary", [False, True]),
    "balance": Attribute("balance", "numeric", None),
    "housing": Attribute("housing", "binary", [False, True]),
    "loan": Attribute("loan", "binary", [False, True]),
    "contact": Attribute("contact", "categorical", ["cellular", "telephone", "unknown"]),
    "day": Attribute("day", "numeric", None),
    "month": Attribute("month", "categorical", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
    "duration": Attribute("duration", "numeric", None),
    "campaign": Attribute("campaign", "numeric", None),
    "pdays": Attribute("pdays", "numeric", None),
    "previous": Attribute("previous", "numeric", None),
    "poutcome": Attribute("poutcome", "categorical", ["failure", "other", "success", "unknown"]),
}

attr_col_map = {
    "age": 0,
    "job": 1,
    "marital": 2,
    "education": 3,
    "default": 4,
    "balance": 5,
    "housing": 6,
    "loan": 7,
    "contact": 8,
    "day": 9,
    "month": 10,
    "duration": 11,
    "campaign": 12,
    "pdays": 13,
    "previous": 14,
    "poutcome": 15,
}

labels = ["yes", "no"]

train_file = "data/hw1/bank/train.csv"
test_file = "data/hw1/bank/test.csv"

def _read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data

def process(file):
    data = _read(file)
    num_cols = len(data[0].strip().split(','))

    S = np.empty((len(data), num_cols), dtype=object)
    for i in range(len(data)):
        S[i] = data[i].strip().split(',')
    return S

def normalizeData(data, attr_col_map, attributes):
    attributes1, data1 = normalizeBinary(data, attr_col_map, attributes)
    attributes2, data2 = convertNumericToBinary(data1, attr_col_map, attributes1)

    return attributes2, data2

def main(): 
    _attributes_normalized, data = normalizeData(process(train_file), attr_col_map, attributes)

    print("attributes:")
    for attr in _attributes_normalized:
        print(_attributes_normalized[attr])

    tree = ID3(data, _attributes_normalized, attr_col_map, maximum_depth=16, IG_algotithm="entropy")

    """
    methods2test = ["entropy", "gini_index", "majority_error"]
    depth2test = np.arange(1, 16)

    print("testing with training data")
    for depth in depth2test:
        print("*****depth:", depth, "*****")
        for method in methods2test:
            print("---| method:", method)
            tree = ID3(data, _attributes_normalized, attr_col_map, maximum_depth=depth, IG_algotithm=method)
            correct_ratio, incorrect_ratio = traverse(tree, data, attr_col_map)
            print("------| correct_ratio:", correct_ratio)

    # test the model with test data
    print("testing with test data")
    test_data = process(test_file)
    """

if __name__ == "__main__":
    main()
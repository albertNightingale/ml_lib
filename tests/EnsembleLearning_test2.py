"""
HW 2 problem 2b
"""
import numpy as np
import copy

from EnsembleLearning.EnsembleLearning import bagging, assess

from ProcessData.Attribute import Attribute
from ProcessData.AttributeNormalizer import convertNumericToBinary
from ProcessData.AttributeNormalizer import normalizeBinary

attributes = {
    "age": Attribute("age", "numeric", None),
    "job": Attribute("job", "categorical", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]),
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

    S = np.empty((len(data), num_cols + 1), dtype=object)
    for i in range(len(data)):
        row_i = data[i].strip().split(',')
        row_i.append(i)
        S[i] = row_i
    return S

# return normalized set of attributes and normalized data
def normalizeData(data, attr_col_map, attributes):
    attributes1, data1 = normalizeBinary(data, attr_col_map, attributes)
    attributes2, data2 = convertNumericToBinary(data1, attr_col_map, attributes1)

    return attributes2, data2


def main(): 
    unprocessed_train_data = copy.deepcopy(process(train_file))
    _attributes_normalized_train, train_data = normalizeData(copy.deepcopy(unprocessed_train_data), attr_col_map, attributes)

    unprocessed_test_data = copy.deepcopy(process(test_file))
    _attributes_normalized_test, test_data = normalizeData(copy.deepcopy(unprocessed_test_data), attr_col_map, attributes)


    T_value_to_test = np.concatenate((np.arange(1, 20), np.arange(50, 501, 50)))

    column_name = "T, train_accuracy, test_accuracy"
    print(column_name)
    for T in T_value_to_test:
        output = str(T) + ","
        classifiers, alpha, error_rates = bagging(train_data, _attributes_normalized_train, attr_col_map, T)
        incorrect_ratio, incorrect_indices = assess(classifiers, alpha, train_data, attr_col_map, _attributes_normalized_train, labels)
        output += str(1-incorrect_ratio) + ","
        incorrect_ratio, incorrect_indices = assess(classifiers, alpha, test_data, attr_col_map, _attributes_normalized_test, labels)
        output += str(1-incorrect_ratio) + "\n"
        output += str(error_rates)
        print(output)
    

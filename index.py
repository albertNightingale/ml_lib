from DecisionTree.ID3 import ID3
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


def __main__(): 
    tree = ID3(train_file, attributes, attr_col_map, maximum_depth=6)
    print("decision tree")
    print(str(tree))

__main__()
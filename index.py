from DecisionTree.ID3 import ID3

attributes = [
    ["vhigh", "high", "med", "low"],
    ["vhigh", "high", "med", "low"],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"]
]

labels = ["unacc", "acc", "good", "vgood"]

train_file = "data/hw1/car/train.csv"

def __main__(): 
    print("hi test")
    tree = ID3(train_file, attributes, labels)
    print(tree)

__main__()
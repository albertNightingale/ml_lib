import numpy as np
from DecisionTree.Node import node
from DecisionTree.InformationGain import IG

# attributes_map is 2D array of attributes values.
# ex:
#       0: [vhigh, high, med, low]
def ID3(file, attributes_map, labels_map):
    data = read(file)
    S = np.empty((len(data), 8), dtype=object)
    for i in range(len(data)):
        arr = data[i].strip().split(',')
        arr.insert(0, i)
        S[i] = arr

    return _ID3(S, S, attributes_map, labels_map)

def read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data


def _ID3(S_before_split, S, attributes_map, labels_map):
    S_labels = S[:, 7]
    # base case
    if np.all(S_labels == S_labels[0]):
        if len(attributes_map) == 0:
            label = findMostCommonLabel(S_before_split)
            return node(attribute=None, S=S, label=label, branch=None)
        else:
            label = S_labels[0]
            return node(attribute=None, S=S, label=label, branch=None)

    # initialize root node
    root_node = node(attribute=None, S=S, label=None, branch={})
    # pick Attribute
    picked_A_index = IG(S, attributes_map)
    A_values = attributes_map[picked_A_index]
    root_node.attribute = picked_A_index 
    attributes_map.pop(picked_A_index) # pop an element

    for v in A_values:
        root_node.branch[v] = None # new branch
        S_v = S[np.where(S[:, picked_A_index] == v)]
        if len(S_v) == 0:
            label = findMostCommonLabel(S)
            root_node.branch[v] = node(attribute=None, S=[], label=label, branch=None)
        else:
            root_node.branch[v] = _ID3(S_before_split, S_v, attributes_map, labels_map)
    return root_node
import numpy as np
import copy

from DecisionTree.Node import node
from DecisionTree.InformationGain import IG
import DecisionTree.util as util

ID3_debug = False

# attributes_map is dictionary.
# ex:
#       name: [vhigh, high, med, low]
def ID3(file, attributes_map, attr_col_map, labels_map):
    data = read(file)
    S = np.empty((len(data), 7), dtype=object)
    for i in range(len(data)):
        S[i] = data[i].strip().split(',')

    return _ID3(S, S, attributes_map, attr_col_map, labels_map)

def read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data


def _ID3(S_before_split, S, attributes_map, attr_col_map, labels_map):
    ID3_debug and print("current size of attribute_map", len(attributes_map))
    S_labels = S[:, 6]
    # base case
    if np.all(S_labels == S_labels[0]):
        ID3_debug and print("find all labels the same:", S_labels[0])
        if len(attributes_map) == 0:
            label = util.findMostCommonLabel(S_before_split)
            return node(attribute=None, S=S, label=label, branch=None) # leaf_node
        else:
            label = S_labels[0]
            return node(attribute=None, S=S, label=label, branch=None) # leaf_node

    # initialize root node
    root_node = node(attribute=None, S=S, label=None, branch={})
    # pick Attribute
    picked_A = IG(S, attributes_map, attr_col_map)
    A_values = attributes_map[picked_A]
    root_node.attribute = picked_A 

    ID3_debug and print("attr_name", picked_A)
    for v in A_values:
        ID3_debug and print("attr_value", v)
        root_node.branch[v] = None # new branch
        S_v = S[np.where(S[:, attr_col_map[picked_A]] == v)]
        if len(S_v) == 0:
            ID3_debug and print("***S_v size is zero")
            label = util.findMostCommonLabel(S)
            root_node.branch[v] = node(attribute=None, S=[], label=label, branch=None) # leaf_node
        else:
            # make a copy of the map and pop it, and pass it to the recursive call later
            attribute_map_copy = copy.deepcopy(attributes_map)
            attribute_map_copy.pop(picked_A) 
            ID3_debug and print("***S_v size is", len(S_v))
            root_node.branch[v] = _ID3(S_before_split, S_v, attribute_map_copy, attr_col_map, labels_map)
    return root_node
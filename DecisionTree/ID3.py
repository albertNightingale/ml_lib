import numpy as np
import copy

from typing import Type

from DecisionTree.Node import node
from DecisionTree.InformationGain import IG
import DecisionTree.util as util
from DecisionTree.Config import config


# attributes_map is dictionary.
# ex:
#       name: [vhigh, high, med, low]
def ID3(file, attributes_map, attr_col_map, maximum_depth=6, IG_algotithm="entropy"):
    data = read(file)
    S = np.empty((len(data), 7), dtype=object)
    for i in range(len(data)):
        S[i] = data[i].strip().split(',')
    
    cfg = config(ID3_debug=False, maximum_depth=maximum_depth, IG_algotithm=IG_algotithm, attr_col_map=attr_col_map, unchanged_S=S)

    return _ID3(cfg, S, attributes_map, 0)

def read(file):
    with open(file, 'r') as f:
        data = []
        for l in f:
            data.append(l)
        return data

def _ID3(cfg: config, S, attributes_map, depth):
    cfg.get_debug() and print("current size of attribute_map", len(attributes_map))
    S_labels = S[:, 6]
    # base case
    if np.all(S_labels == S_labels[0]):
        cfg.get_debug() and print("find all labels the same:", S_labels[0])
        if len(attributes_map) == 0:
            label = util.findMostCommonLabel(cfg.get_unchanged_S())
            return node(attribute=None, S=S, label=label, branch=None) # leaf_node
        else:
            label = S_labels[0]
            return node(attribute=None, S=S, label=label, branch=None) # leaf_node

    # initialize root node
    root_node = node(attribute=None, S=S, label=None, branch={})
    # pick Attribute
    picked_A = IG(S, attributes_map, cfg.get_attr_col_map(), cfg.get_IG_algotithm())
    A_values = attributes_map[picked_A]
    root_node.attribute = picked_A 

    cfg.get_debug() and print("attr_name", picked_A)
    for v in A_values:
        cfg.get_debug() and print("attr_value", v)
        root_node.branch[v] = None # new branch
        S_v = S[np.where(S[:, cfg.get_attr_col_map()[picked_A]] == v)]
        if len(S_v) == 0:
            cfg.get_debug() and print("***S_v size is zero")
            label = util.findMostCommonLabel(S)
            root_node.branch[v] = node(attribute=None, S=[], label=label, branch=None) # leaf_node
        else:
            cfg.get_debug() and print("***S_v size is", len(S_v))
            # make a copy of the map and pop it, and pass it to the recursive call later
            attribute_map_copy = copy.deepcopy(attributes_map)
            attribute_map_copy.pop(picked_A) 
            root_node.branch[v] = _ID3(cfg, S_v, attribute_map_copy, depth)
    return root_node
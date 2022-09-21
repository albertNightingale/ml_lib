import numpy as np
import copy

from typing import Type

from DecisionTree.Node import node
from DecisionTree.InformationGain import IG
import DecisionTree.util as util
from DecisionTree.Config import config

# import attributes
import ProcessData.Attribute as Attribute

# attr_dict is a dictionary of attributes.
# ex:
#       name: [vhigh, high, med, low]
# 
def ID3(S, attr_dict, attr_col_map, maximum_depth=6, IG_algotithm="entropy"):    
    num_columns = len(S[0])
    cfg = config(ID3_debug=False, column_count=num_columns, maximum_depth=maximum_depth, IG_algotithm=IG_algotithm, attr_col_map=attr_col_map, unchanged_S=S)

    return _ID3(cfg, S, attr_dict, 0)



def _ID3(cfg: config, S, attr_dict, depth):
    cfg.get_debug() and print("!!!!current size of attribute_map", len(attr_dict))
    cfg.get_debug() and print("!!!!current size of S", len(S))
    S_labels = S[:, cfg.get_label_column()]
    # base case # 1
    if np.all(S_labels == S_labels[0]):
        cfg.get_debug() and print("!!!! BASE CASE # 1: find all labels the same:", S_labels[0])
        label = S_labels[0]
        if len(attr_dict) == 0: # no more attributes to split
            label = util.findMostCommonLabel(cfg.get_unchanged_S(), cfg)
            return node(attribute=None, S=S, label=label, branch=None) # leaf_node
        return node(attribute=None, S=S, label=label, branch=None) # leaf_node
    # base case # 2
    if len(attr_dict) == 0: # no more attributes to split
        cfg.get_debug() and print("!!!! BASE CASE # 2: no more attributes to split")
        label = util.findMostCommonLabel(cfg.get_unchanged_S(), cfg)
        return node(attribute=None, S=S, label=label, branch=None) # leaf_node

    # initialize root node
    root_node = node(attribute=None, S=S, label=None, branch={})
    # pick Attribute
    picked_A = IG(S, attr_dict, cfg.get_attr_col_map(), cfg)
    A = attr_dict[picked_A]
    root_node.attribute = picked_A 

    cfg.get_debug() and print("attr_name", picked_A)

    # iterate over each value of A and generate a subtree for each value
    for v in A.get_values():
        cfg.get_debug() and print("attr_value", v)
        root_node.branch[v] = None # new branch
        S_v = S[S[:, cfg.get_attr_col_map()[picked_A]] == v]
        if len(S_v) == 0:
            cfg.get_debug() and print("***S_v size is zero")
            label = util.findMostCommonLabel(S, cfg)
            root_node.branch[v] = node(attribute=None, S=[], label=label, branch=None) # leaf_node
        else:
            cfg.get_debug() and print("***S_v size is", len(S_v))
            if depth < cfg.get_maximum_depth():
                # make a copy of the map and pop it, and pass it to the recursive call later
                attribute_map_copy = copy.deepcopy(attr_dict)
                attribute_map_copy.pop(picked_A) 
                root_node.branch[v] = _ID3(cfg, S_v, attribute_map_copy, depth+1)
            else:
                cfg.get_debug() and print("***reached maximum depth level")
                label = util.findMostCommonLabel(S, cfg)
                root_node.branch[v] = node(attribute=None, S=[], label=label, branch=None) # leaf_node

    return root_node


def traverse(root_node: node, dataset, attr_col_map, attr_map):
    correct = 0
    incorrect = 0
    total = len(dataset)
    num_columns = len(dataset[0])
    for data in dataset:
        attr = data[:num_columns-1]
        label = data[num_columns-1:num_columns]
        if _traverse(root_node, attr, label, attr_col_map, attr_map):
            correct += 1
        else:
            incorrect += 1
    return correct/total, incorrect/total

def _traverse(nd: node, attr, label, attr_col_map, attr_map):
    if nd.label != None:
        if nd.label != label: 
            return False
        else: 
            return True
    else:
        attr_idx = attr_col_map[nd.attribute]
        attr_obj = attr_map[nd.attribute]

        test_attr_val = attr[attr_idx]

        if attr_obj.get_type() == "numeric":
            converted_attr_val = None
            if float(test_attr_val) <= attr_obj.get_median():
                converted_attr_val = attr_obj.get_values()[0]
            else:
                converted_attr_val = attr_obj.get_values()[1]
            return _traverse(nd.branch[converted_attr_val], attr, label, attr_col_map, attr_map)
        elif attr_obj.get_type() == "binary":
            converted_attr_val = None
            if test_attr_val == "Yes":
                converted_attr_val = True
            else:
                converted_attr_val = False
            return _traverse(nd.branch[converted_attr_val], attr, label, attr_col_map, attr_map)
        else:
            return _traverse(nd.branch[test_attr_val], attr, label, attr_col_map, attr_map)

    



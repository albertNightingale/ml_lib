
class node:
    def __init__(self, attribute=None, S=None, label=None, branch={}):
        self.attribute = attribute
        self.S = S
        self.label = label
        self.branch = branch

    def __str__(self):        
        node_str = ""
        for attr_value in self.branch:
            attr_node = self.branch[attr_value]
            attribute_str = "attribute_idx_"+str(self.attribute) + "==" + attr_value + "\n"
            node_str += attribute_str
            node_str += attr_node.to_str(1, attr_value)

        return node_str
            
    def to_str(self, depth, attr_value):
        if self.label != None:
            return node._str_tabs(depth) + "label==" + self.label + "\n"
        
        # look at the subbranches
        node_str = ""
        attribute_str = node._str_tabs(depth) + " attribute_idx_"+str(self.attribute) + "==" + attr_value + "\n"
        node_str += attribute_str
        # look at it sub branches
        for feature_value in self.branch:
            feature_node = self.branch[feature_value]
            node_str += feature_node.to_str(depth+1, feature_value)
        return node_str

    def _str_tabs(depth):
        s = ""
        for i in range(depth):
            s += "|   "
        return s
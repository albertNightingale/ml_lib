
class node:
    def __init__(self, attribute=None, S=None, label=None, branch={}, prev_attribute=None, prev_attribute_val=None):
        self.attribute = attribute
        self.S = S
        self.label = label
        self.branch = branch
        self.prev_attribute = prev_attribute
        self.prev_attribute_val = prev_attribute_val



class ada_config: 
    def __init__(self, unmodified_S, ada_debug=False, iterations = None, attr_col_map=None, attr_dict=None):
        self.ada_debug = ada_debug
        self.iterations = iterations
        self.attr_col_map = attr_col_map
        self.unmodified_S = unmodified_S
        self.attr_dict = attr_dict

    def get_normal_weight(self):
        return 1/len(self.unmodified_S)
    
    def get_index_column(self):
        return self.unmodified_S.shape[1] - 1
    
    def get_label_column(self):
        return self.unmodified_S.shape[1] - 2

    def get_column_count(self):
        return self.unmodified_S.shape[1]

    def get_debug(self):
        return self.ada_debug

    def get_iterations(self):
        return self.iterations

    def get_attr_col_map(self):
        return self.attr_col_map

    def get_attr_dict(self):
        return self.attr_dict

    def get_unmodified_S(self):
        return self.unmodified_S

class bagging_config: 
    def __init__(self, unmodified_S, bagging_debug=False, iterations = None, attr_col_map=None, attr_dict=None):
        self.bagging_debug = bagging_debug
        self.iterations = iterations
        self.attr_col_map = attr_col_map
        self.unmodified_S = unmodified_S
        self.attr_dict = attr_dict

    def get_normal_weight(self):
        return 1/len(self.unmodified_S)
    
    def get_index_column(self):
        return self.unmodified_S.shape[1] - 1
    
    def get_label_column(self):
        return self.unmodified_S.shape[1] - 2

    def get_column_count(self):
        return self.unmodified_S.shape[1]

    def get_debug(self):
        return self.bagging_debug

    def get_iterations(self):
        return self.iterations

    def get_attr_col_map(self):
        return self.attr_col_map

    def get_attr_dict(self):
        return self.attr_dict

    def get_unmodified_S(self):
        return self.unmodified_S
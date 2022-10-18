

class config: 
    def __init__(self, unmodified_S, weight=None, ID3_debug=False, maximum_depth=6, IG_algotithm="entropy", attr_col_map=None):
        self.weight = weight
        self.ID3_debug = ID3_debug
        self.maximum_depth = maximum_depth
        self.IG_algotithm = IG_algotithm
        self.attr_col_map = attr_col_map
        self.unmodified_S = unmodified_S

    def get_normal_weight(self):
        return 1/len(self.unmodified_S)

    def get_weight(self):
        return self.weight
    
    def get_index_column(self):
        return self.unmodified_S.shape[1] - 1
    
    def get_label_column(self):
        return self.unmodified_S.shape[1] - 2

    def get_column_count(self):
        return self.unmodified_S.shape[1]

    def get_debug(self):
        return self.ID3_debug

    def get_maximum_depth(self):
        return self.maximum_depth
        
    def get_IG_algotithm(self):
        return self.IG_algotithm

    def get_attr_col_map(self):
        return self.attr_col_map

    def get_unmodified_S(self):
        return self.unmodified_S
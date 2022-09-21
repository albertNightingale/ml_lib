

class config: 
    def __init__(self, ID3_debug=False, column_count=0, maximum_depth=6, IG_algotithm="entropy", attr_col_map=None, unchanged_S=None):
        self.ID3_debug = ID3_debug
        self.maximum_depth = maximum_depth
        self.column_count = column_count
        self.label_column = column_count - 1
        self.IG_algotithm = IG_algotithm
        self.attr_col_map = attr_col_map
        self.unchanged_S = unchanged_S
    
    def get_label_column(self):
        return self.label_column

    def get_column_count(self):
        return self.column_count

    def get_debug(self):
        return self.ID3_debug

    def get_maximum_depth(self):
        return self.maximum_depth
        
    def get_IG_algotithm(self):
        return self.IG_algotithm

    def get_attr_col_map(self):
        return self.attr_col_map

    def get_unchanged_S(self):
        return self.unchanged_S
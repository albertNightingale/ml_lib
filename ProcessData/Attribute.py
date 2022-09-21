

class Attribute:
    def __init__(self, name, category, values):
        self.name = name
        self.type = category
        self.values = values
        self.median = None

    def set_median(self, median):
        self.median = median
    
    def get_median(self):
        return self.median

    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.type

    def get_values(self):
        return self.values

    def set_values(self, values):
        self.values = values

    def __str__(self):
        return "Attribute: " + self.name + " type: " + self.type + " values: " + str(self.values)
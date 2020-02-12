# Base types for MC interface
# from enum import Enum

class ConstraintType: #(Enum):
    # EQUALITY = 0
    # LESS_EQ = 1
    # LESS = 2
    # GREATER_EQ = 3
    # GREATER = 4
    type2str = {
        'EQUALITY': "=",
        'LESS_EQ': "<=",
        'LESS' : "<",
        'GREATER_EQ' : ">=",
        'GREATER' : ">"
    }
    def __init__(self, type_):
        assert(type_ in self.type2str.keys())
        self._type = type_
    def __repr__(self): 
        return self.type2str[self._type]

class Constraint:
    def __init__(self, eqtype: ConstraintType):
        """
        sum_i(monomial_i) ConstraintType scalar
        e.g. 5x + 3y <= 0
        """
        self.type = eqtype
        self.monomials = [] # list of tuples of (coeff, var)
        self.scalar = 0
    
    def __repr__(self):
        out = ""
        if len(self.monomials) > 0:
            first_m = self.monomials[0]
            out += str(first_m[0]) + "*" + str(first_m[1])
        for i in range(1,len(self.monomials)):
            m = self.monomials[i]
            out += " + " + str(m[0]) + "*" + str(m[1])
        out += " " + self.type.__repr__() + " "
        out += str(self.scalar)
        return out



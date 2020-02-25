# Base types for MC interface
# from enum import Enum
import numpy as np

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

    # note: __eq__ and __hash__ are required for making a dictionary where
    # the keys are of type ConstraintType
    def __eq__(self, other):
        return self._type == other._type

    def __hash__(self):
        return hash(self._type)

    def __repr__(self): 
        return self.type2str[self._type]

class Monomial:
    def __init__(self, coeff, var):
        assert(isinstance(coeff, int) or isinstance(coeff, float))
        assert(isinstance(var, str))
        self.coeff = coeff
        self.var = var

class Constraint:
    def __init__(self, eqtype: ConstraintType, monomials=[], scalar = 0):
        """
        sum_i(monomial_i) ConstraintType scalar
        e.g. 5x + 3y <= 0
        """
        self.type = eqtype
        self.monomials = monomials 
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

class MatrixConstraint:
    """
    Akin to a Constraint, but for constraints of the form Ax R b,
    where R represents a relation such as =, >=, <=, >, <.
    @pre If dim(A) = (m,n), dim(x) = n , dim(b) = m 
    """
    def __init__(self, eqtype: ConstraintType, A=np.array([[]]), x = np.array([[]]), b = np.array([[]])):
        self.type = eqtype
        self.A = A # contains real numbers
        self.x = x # contains variables
        self.b = b # contains real numbers
    
    def __repr__(self):
        s = "<Constraint: Ax" + self.type.__repr__() + "b>\n"
        s += "  {A: " + np.array(self.A).__repr__() + "\n" 
        s += "  x: " + np.array(self.x, dtype=object).__repr__() +"\n"
        s += "  b: " + np.array(self.b).__repr__() + "}\n"
        return s

class ReluConstraint():
    """
    varout = relu(varin)
    """
    def __init__(self, varin=None, varout=None):
        self.varin = varin
        self.varout = varout
    def __repr__(self):
        return "<Constraint: " + str(self.varout) + " = relu(" + str(self.varin) + ") >\n"

def matrix_to_scalar(c : MatrixConstraint):
    """
    Takes a MatrixConstraint and returns a list of Constraint 
    """
    pass
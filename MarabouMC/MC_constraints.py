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
        assert(np.isreal(coeff)) 
        assert(isinstance(var, str))
        self.coeff = coeff
        self.var = var

class Constraint:
    def __init__(self, ctype: ConstraintType, monomials=[], scalar = 0):
        """
        sum_i(monomial_i) ConstraintType scalar
        e.g. 5x + 3y <= 0
        """
        self.type = ctype
        self.monomials = monomials 
        self.scalar = scalar
    
    def complement(self):
        # return complement of constraint
        if self.type == ConstraintType('GREATER'):
            ccomp = Constraint(ConstraintType('LESS_EQ'))
        elif self.type == ConstraintType('LESS'):
            ccomp = Constraint(ConstraintType('GREATER_EQ'))
        elif self.type == ConstraintType('GREATER_EQ'):
            print("Warning: Can solver handle strict inequalities?")
            ccomp = Constraint(ConstraintType('LESS'))
        elif self.type == ConstraintType('LESS_EQ'):
            print("Warning: Can solver handle strict inequalities?")
            ccomp = Constraint(ConstraintType('GREATER'))
        else:
            ccomp = None
            raise NotImplementedError
        ccomp.monomials = self.monomials
        ccomp.scalar = self.scalar
        return ccomp
    
    def __repr__(self):
        out = ""
        if len(self.monomials) > 0:
            first_m = self.monomials[0]
            out += str(first_m.coeff) + "*" + str(first_m.var)
        for i in range(1,len(self.monomials)):
            m = self.monomials[i]
            out += " + " + str(m.coeff) + "*" + str(m.var)
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
        # add assertions for the precondition
        self.type = eqtype
        self.A = A # contains real numbers
        self.x = x # contains variables. # when parsed from tf, will be numpy arrays.
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

class MaxConstraint():
    """
    varout = max(var1in, var2in)
    """
    def __init__(self, var1in, var2in, varout):
        self.var1in = var1in
        self.var2in = var2in
        self.varout = varout
    def __repr__(self):
        return "<Constraint: " + str(self.varout) + " = max(" + str(self.var1in) + " , " + str(self.var2in) + ") >\n"   

def matrix_to_scalar(c : MatrixConstraint):
    """
    Takes a MatrixConstraint and returns a list of Constraint s
    """
    # form: Ax R b
    # for every row in A, add a Marabou 'equation'
    scalar_constraints = []
    for row in range(c.A.shape[0]):
        coefficients = c.A[row, :]
        scalar = c.b[row]
        # construct monomials list 
        monomials = [Monomial(c,v) for c,v in zip(coefficients, c.x)]
        scalar_constraints.append(Constraint(c.type, monomials=monomials, scalar=scalar))
    return scalar_constraints



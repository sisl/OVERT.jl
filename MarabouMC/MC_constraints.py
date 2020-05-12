# Base types for MC interface
# from enum import Enum
import numpy as np
import copy

class ConstraintType:
    type2str = {
        'EQUALITY': "=",
        'NOT_EQUAL': "not=",
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
    def __neg__(self):
        """Unary negation"""
        return Monomial(-self.coeff, self.var)

    def __repr__(self):
        return str(self.coeff) + "*'" + str(self.var) + "'"

class AbstractConstraint:
    def __init__(self):
        self.type_complement = {
            ConstraintType('GREATER'): ConstraintType('LESS_EQ'),
            ConstraintType('LESS_EQ'): ConstraintType('GREATER'),
            ConstraintType('LESS'): ConstraintType('GREATER_EQ'),
            ConstraintType('GREATER_EQ'): ConstraintType('LESS'),
            ConstraintType('EQUALITY') : ConstraintType('NOT_EQUAL'),
            ConstraintType('NOT_EQUAL'): ConstraintType('EQUALITY')
        }

class Constraint(AbstractConstraint):
    def __init__(self, ctype: ConstraintType, monomials=[], scalar = 0):
        """
        A class to represent linear constraints.
        sum_i(monomial_i) ConstraintType scalar
        e.g. 5x + 3y <= 0
        """
        super().__init__()
        if isinstance(ctype, str):
            self.type = ConstraintType(ctype)
        elif isinstance(ctype, ConstraintType):
            self.type = ctype
        self.monomials = monomials 
        self.scalar = scalar
    
    def complement(self):
        ccomp = Constraint(self.type_complement[self.type])
        ccomp.monomials = self.monomials
        ccomp.scalar = self.scalar
        return ccomp

    def get_geq(self):
        """
        Return a equivalent version of this constraint with >=
        as the relational operator.
        """
        geq_c = Constraint(ConstraintType('GREATER_EQ'))
        if self.type == ConstraintType('LESS_EQ'):
            # flip all coeff signs and scalar sign and change to >=
            geq_c.monomials = [-m for m in self.monomials]
            geq_c.scalar = -self.scalar
        elif self.type == ConstraintType('GREATER_EQ'):
            geq_c = copy.deepcopy(self)
        else: # asking for conversion from strict ineq to non-strict ineq
            # if complement, we want to go from e.g.
                # 5x + 6y < 6 
                # to 5x + 6y <= 6.001
            # if regular assertion (not inverted) we want to go from e.g.
                # 5x + 6y < 6
                # to 5x + 6y <= 5.999
            raise NotImplementedError
        
        return geq_c

    
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

class MatrixConstraint(AbstractConstraint):
    """
    Akin to a Constraint, but for constraints of the form Ax R b,
    where R represents a relation such as =, >=, <=, >, <.
    @pre If dim(A) = (m,n), dim(x) = n , dim(b) = m 
    """
    def __init__(self, ctype: ConstraintType, A=np.zeros((0,0)), x = np.zeros((0,0)), b = np.zeros((0,0))):
        super().__init__()
        if isinstance(ctype, str):
            self.type = ConstraintType(ctype)
        elif isinstance(ctype, ConstraintType):
            self.type = ctype
        # assertions for the precondition
        assert(A.shape[1] == x.shape[0])
        assert(A.shape[0] == b.shape[0])
        self.A = A # contains real numbers
        self.x = x # contains variables. # when parsed from tf, will be numpy arrays.
        self.b = b # contains real numbers
    
    def complement(self):
        """ 
        Return complement of this constraint as list in DNF.
        """
        scalar_constraints = matrix_to_scalar(self)
        complements = []
        for c in scalar_constraints:
            complements.append(c.complement())
        return complements
    
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
    def __init__(self, varsin, varout):
        self.var1in = varsin[0]
        self.var2in = varsin[1]
        self.varout = varout
    def __repr__(self):
        return str(self.varout) + " = max(" + str(self.var1in) + " , " + str(self.var2in) + ")"


def matrix_to_scalar(c : MatrixConstraint):
    """
    Takes a MatrixConstraint and returns a list of Constraint s
    """
    # form: Ax R b
    # for every row in A, add a Marabou 'equation'
    scalar_constraints = []
    for row in range(c.A.shape[0]):
        coefficients = c.A[row, :]
        scalar = c.b[row][0] # assumes 1D b
        # construct monomials list 
        monomials = [Monomial(c,v) for c,v in zip(coefficients, c.x.flatten())]
        scalar_constraints.append(Constraint(c.type, monomials=monomials, scalar=scalar))
    return scalar_constraints

class NLConstraint(AbstractConstraint):
    def __init__(self, ctype: ConstraintType, left, right, indep_var):
        """
        A class to represent 1-D nonlinear constraints.
        left R right
        left is a variable string and right is a string holding a nonlinear expression.
        e.g. left = "x56"
             right = "sin(x86)"
        indep_var is the independent variable in the string / nonlinear expression.
        where R is a relation in the set: <, <=, >, >=, =
        """
        super().__init__()
        if isinstance(ctype, str):
            self.type = ConstraintType(ctype)
        elif isinstance(ctype, ConstraintType):
            self.type = ctype
        self.left = left
        self.right = right
        self.indep_var = indep_var
    
    def complement(self):
        return NLConstraint(self.type_complement[self.type], self.left, self.right, self.indep_var)
    
    def __repr__(self):
        s = "<Non Linear Constraint: left " + self.type.__repr__() + " right\n"
        return s

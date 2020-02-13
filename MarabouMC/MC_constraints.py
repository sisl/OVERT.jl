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

class MatrixConstraint:
    """
    Akin to a Constraint, but for constraints of the form Ax <= b.
    @pre @post If dim(A) = (m,n), dim(x) = n , dim(b) = m 
    """
    def __init__(self, eqtype: ConstraintType):
        self.type = eqtype
        self.A = [[]]
        self.x = []
        self.b = []
    
    def __repr__(self):
        """
        Pretty printing Ax R b
        """
        out = ""
        # print A matrix
        out += np.array(self.A).__repr__()
        # print x
        out = out.split('\n')
        out[0] += " ["
        for i in range(min(len(self.x), len(out))):
            if i == 0:
                out[i]+= self.x[i]
            elif i == len(out)-1:
                out[i]+= ' '+self.x[i]
            else:
                out[i]+= '  '+self.x[i]
        i+=1
        while i < len(out): ## if len(out) > len(x)
            out[i] += '  '
            i+=1
        while i < len(self.x): ## if len(x) > len(out)
            out.append((len(out[-1])-1)*' ' + self.x[i])
            i+=1
        out[len(self.x)-1] += "]"
        # print relation
        mrow = int(np.floor((len(out) -1)/2))
        out[mrow] += "  " + self.type.__repr__() + "  "
        out = [out[i]+"      " if not i==mrow else out[i] for i in range(len(out)) ]
        # print constant/scalar
        out[0] += "["
        for j in range(len(self.b)):
            if not j==0:
                out[j] += " "+str(self.b[j])
            else:
                out[j] += str(self.b[j])
        out[len(self.b) - 1] += "]"
        return '\n'.join(out)

class Relu():
    """
    varout = relu(varin)
    """
    def __init__(self, varin, varout):
        self.varin = varin
        self.varout = varout
    def __repr__(self):
        return str(self.varout) + " = relu(" + str(self.varin) + ")\n"

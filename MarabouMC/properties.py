# property classes
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint


# property
class Property():
    """
    Property that you want to hold.
    For now, box set over state variables.
    Future: linear constraints on outputs OR box set.
    """
    def __init__(self, arg):
        pass

    def complement(self):
        # return complement of desired property
        pass

# property
class ConstraintProperty():
    """
    Property that you want to hold. A list of constraints in CNF.
    """
    def __init__(self, c):
        self.constraints = c
        self.constraint_complements = []

    def complement(self):
        # return complement of desired property
        # for now, only support single simple constraint
        assert(len(self.constraints) == 1)
        c = self.constraints[0]
        # perhaps we should implement the complement method in the constraint class
        if c.type == ConstraintType('GREATER'):
            ccomp = Constraint(ConstraintType('LESS_EQ'))
        elif c.type == ConstraintType('LESS'):
            ccomp = Constraint(ConstraintType('GREATER_EQ'))
        elif c.type == ConstraintType('GREATER_EQ'):
            ccomp = Constraint(ConstraintType('LESS_EQ'))
        elif c.type == ConstraintType('LESS_EQ'):
            ccomp = Constraint(ConstraintType('GREATER'))
        else:
            ccomp = None
            raise NotImplementedError
        ccomp.monomials = c.monomials
        ccomp.scalar = c.scalar
        self.constraint_complements = [ccomp]
        # for future, a code sketch:
        # complements = []
        # for c in self.constraints:
        #     complement.append(self.negate(c)) 
        # then disjunct all the complements using MaxConstraint

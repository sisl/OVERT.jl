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
        # for future, a code sketch:
        complements = []
        # handle first constraint first, to start disjunct train
        for c in self.constraints:
            # take complement, and turn into geq ineq
            complements.append(c.complement().get_geq())
            # then disjunct all the complements using MaxConstraint
        
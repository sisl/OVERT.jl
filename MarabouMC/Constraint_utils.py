# constraint utils

import numpy as np
from MC_constraints import ConstraintType, Constraint, MatrixConstraint, Monomial, matrix_to_scalar

def matrix_equality_constraint(varsin, varsout):
    """
    preconditions: 
    varsin, varsout are vertical vectors (nx1), of type 2D np array or similar
    """
    # x = y (want in form Az = b)
    # z := [x; y]
    # A = [I, -I]
    # b = 0
    # [I, -I][x; y] = 0  -> x - y = 0
    A = np.hstack((np.eye(len(varsin)), -np.eye(len(varsout))))
    z = np.vstack((varsin, varsout))
    b = np.zeros((A.shape[0],1))
    return MatrixConstraint(ConstraintType('EQUALITY'), A=A, x=z, b=b)

def equality_constraint(varsin, varsout):
    """
    If you need a list of scalar constraints instead of a single matrix constraint.
    """
    assert(len(varsin) == len(varsout))
    if len(varsin) > 1:
        mc = matrix_equality_constraint(varsin, varsout)
        return matrix_to_scalar(mc)
    else:
        mono1 = Monomial(1, varsin[0][0])
        mono2 = Monomial(-1, varsout[0][0])
        return Constraint(ConstraintType('EQUALITY'), monomials=[mono1, mono2], scalar=0)



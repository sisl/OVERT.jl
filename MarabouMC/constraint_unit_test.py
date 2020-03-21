# tests for constraints
# unit testing
import numpy as np
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, Monomial, matrix_to_scalar

c = Constraint(ConstraintType('LESS_EQ'))
c.monomials.append(Monomial(3.1, 'x'))
c.monomials.append(Monomial(1.4, 'y'))
c.monomials.append(Monomial(9.2, 'z'))
print(c)

c = MatrixConstraint(ConstraintType('LESS_EQ'))
c.A = np.array([[4,5,6],[6,7,8]])
c.x = np.array([['x'], ['y'], ['z']])
c.b = np.array([[1],[2]])
print(c)

c = MatrixConstraint(ConstraintType('EQUALITY'))
c.A = np.array([[4,5],[6,7], [8,9]])
c.x = np.array([['x'], ['y']])
c.b = np.array([[1],[2],[3]])
print(c)

c = MatrixConstraint(ConstraintType('EQUALITY'))
c.A = np.array([[4,5],[6,7]])
c.x = np.array([['x'], ['y']])
c.b = np.array([[1],[2]])
print(c)

cs = matrix_to_scalar(c)
print(cs)

print(c.complement())

print("All tests pass.")
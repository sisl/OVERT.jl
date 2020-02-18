# tests for constraints
# unit testing

from MC_constraints import Constraint, ConstraintType, MatrixConstraint

c = Constraint(ConstraintType('LESS_EQ'))
c.monomials.append((3.1, 'x'))
c.monomials.append((1.4, 'y'))
c.monomials.append((9.2, 'z'))
print(c)

c = MatrixConstraint(ConstraintType('LESS_EQ'))
c.A = [[4,5,6],[6,7,8]]
c.x = ['x', 'y', 'z']
c.b = [1,2]
print(c)

c = MatrixConstraint(ConstraintType('EQUALITY'))
c.A = [[4,5],[6,7], [8,9]]
c.x = ['x', 'y']
c.b = [1,2,3]
print(c)

c = MatrixConstraint(ConstraintType('EQUALITY'))
c.A = [[4,5],[6,7]]
c.x = ['x', 'y']
c.b = [1,2]
print(c)
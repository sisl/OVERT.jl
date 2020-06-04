from gurobi_interface import GurobiPyWrapper
from MC_constraints import MatrixConstraint, Constraint, ReluConstraint, MaxConstraint, ConstraintType, Monomial
import numpy as np
import colored_traceback.always

c1 = Constraint('EQUALITY', monomials=[Monomial(1., "theta@0"), Monomial(-2., "x@1")], scalar=1.3)
c2 = MatrixConstraint('LESS_EQ', A=np.random.rand(3,3), x = np.array(['x','theta', 'x2']), b=np.random.rand(3,))
c3 = MaxConstraint(varsin=['t','b'], varout='theta')
c4 = ReluConstraint(varin='bob', varout='alice')

solver = GurobiPyWrapper()

solver.assert_init({'theta':[7,9], 'bob':[-10,1]})

solver.assert_constraints([c1,c2,c3,c4])

result, vals, stats = solver.check_sat()
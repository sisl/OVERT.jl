from gurobi_interface import GurobiPyWrapper
from MC_constraints import MatrixConstraint, Constraint, ReluConstraint, MaxConstraint, ConstraintType, Monomial
import numpy as np
import colored_traceback.always

def feasible_problem():
    c1 = Constraint('EQUALITY', monomials=[Monomial(1., "theta@0"), Monomial(-2., "x@1")], scalar=1.3)
    c2 = MatrixConstraint('LESS_EQ', A=np.random.rand(3,3), x = ["x","theta", "x2"], b=np.random.rand(3,))
    # NOTE: FOR GUROBI INTERFACE, variable stringnames in x must be in a list NOT a numpy array, inside MatrixConstraint
    c3 = MaxConstraint(varsin=['t','b'], varout="theta")
    c4 = ReluConstraint(varin="bob", varout="alice")

    solver = GurobiPyWrapper()

    solver.assert_init({'theta':[7,9], 'bob':[-10,1]})

    solver.assert_constraints([c1,c2,c3,c4])

    result, vals, stats = solver.check_sat()

def infeasible_problem():
    c1 = Constraint('LESS_EQ', monomials=[Monomial(1.,"x")], scalar=5)
    c2 = Constraint('GREATER_EQ', monomials=[Monomial(1.,"x")], scalar=10)
    solver = GurobiPyWrapper()
    solver.assert_constraints([c1,c2])
    result, vals, stats = solver.check_sat()

def infeasible_problem_relu():
    c1 = ReluConstraint(varin="x",varout="y")
    c2 = Constraint("LESS_EQ", monomials=[Monomial(1.,"y")], scalar=-5)
    solver = GurobiPyWrapper()
    solver.assert_constraints([c1,c2])
    result, vals, stats = solver.check_sat()
    print("result is:", result)

feasible_problem()
infeasible_problem()
infeasible_problem_relu()
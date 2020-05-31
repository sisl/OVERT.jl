from MC_constraints import *
from gurobipy import Model
import numpy as np

class MarabouWrapper():
    def __init__(self, n_worker=4):
        # initialize "clean" query
        # self.clear()
        self.mip_model = Model()
        self.mip_var_dict = {}

    def get_mip_var(self, v):
        if v in self.mip_var_dict:
            return self.mip_var_dict[v]
        else:
            if


    def assert_constraints(self, constraints):
        # add constraints to Marabou IPQ object
        for c in constraints:
            self.assert_one_constraint(c)

    def assert_one_constraint(self, constraint):
        if isinstance(constraint, MatrixConstraint):
            self.assert_matrix_constraint(constraint)
        elif isinstance(constraint, Constraint):
            self.assert_simple_constraint(constraint)
        elif isinstance(constraint, ReluConstraint):
            self.assert_relu_constraint(constraint)
        elif isinstance(constraint, MaxConstraint):
            self.assert_max_constraint(constraint)
        else:
            raise NotImplementedError


    def assert_simple_constraint(self, constraint):
        coeffs = [m.coeff for m in constraint.monomials]
        constraint_vars = [m.var for m in constraint.monomials]
        marabou_vars = self.get_new_vars(constraint_vars)
        self.add_marabou_eq(coeffs, marabou_vars, constraint.type, constraint.scalar)


if __name__ == "__main__":
    solver = MarabouWrapper()
    c = Constraint("EQUALITY", monomials=[Monomial(1.0, "x1"), Monomial(1.0, "x2")], scalar=0)
    solver.assert_constraints(c)
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint, MaxConstraint
import numpy as np
from MC_interface import Result
import gurobipy as gp
import pickle
import os

"""
Plan:
- implement necessary high level functions for use with BMC:
    - clear - DONE
    - assert init set - DONE
    - assert constraints
    - check_sat

- low level helper functions:
    - get_new_var - DONE
    - assert 1 constraint
        - assert each type of constraint: relu, max, affine 1D, affine nD (matrix) 

- For relu and max, implement MIPverify encoding
    - go get encoding from paper
    - implement
- Design of wrapper:
    - data members:
        - gurobi model
        - varmap, BMC vars - > gurobi vars

Useful references:
https://www.gurobi.com/documentation/8.1/quickstart_mac/py_example_mip1_py.html

"""

class GurobiPyWrapper()
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.model = gp.Model("OVERT")
        self.var_map = {}
    
    def assert_init(self, init_set):
        """ assert states in InitSet set
        for now, assume set is a box set over inputs
        of the form: {"x@0": (0, 5), "theta@0": (-np.pi/4, np.pi/4)}
        """
        for k in init_set.keys():
            self.get_new_var(k, gp.GRB.CONTINUOUS, lb=init_set[k][0], ub=init_set[k][1])

    def get_new_var(self, name, vtype, lb=-gp.GRB.INFINITY, ub = gp.GRB.INFINITY):
        """
        type should be one of:
        GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
        """
        if name not in self.var_map.keys():
            gbv = self.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
            self.var_map[name] = gbv
        return self.var_map[name]
    
    def assert_constraints(self, constraints):
        pass

    def assert_1_constraint(self, c):
        pass





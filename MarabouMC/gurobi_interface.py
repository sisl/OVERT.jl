from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint, MaxConstraint
import numpy as np
from MC_interface import Result
import gurobipy as gp
inf = gp.GRB.INFINITY 
import pickle
import os
from MC_interface import substitute_c

"""
Plan:
- implement necessary high level functions for use with BMC:
    - clear - DONE
    - assert init set - DONE
    - assert constraints - DONE
    - check_sat - DONE

- low level helper functions:
    - get_new_var(s) - DONE, DONE
    - assert 1 constraint - DONE
        - assert each type of constraint: relu - done, max - done, affine 1D - DONE, affine nD (matrix) - DONE

- For relu and max, implement MIPverify encoding
    - go get encoding from paper
    - implement
- Design of wrapper:
    - data members:
        - gurobi model
        - varmap, BMC vars - > gurobi vars
    - incorporate any bounds calculated by overt
        - can init solver with dictionary -- if passed variable with same prefix...
        (stripped of timing suffix like '@xx') then when doing "add var" in gurobi,
        add bounds for this variable...? OH but...over time...bounds will change...
        TABLE til figure out multi-step calls to overt.


Useful references:
https://www.gurobi.com/documentation/8.1/quickstart_mac/py_example_mip1_py.html

"""

class GurobiPyWrapper():
    def __init__(self, PWL_encoding='native'):
        self.clear()
        # encoding for PWL disjunctions like max, relu
        self.PWL_encoding = PWL_encoding # may be either 'native' or 'MIPverify'
    
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

    def get_new_var(self, name, vtype, lb=-inf, ub =inf):
        """
        type should be one of:
        GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
        """
        if name not in self.var_map.keys():
            gbv = self.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
            self.var_map[name] = gbv
        return self.var_map[name]
    
    def handle_args(self, names, vtypes, lbs, ubs):
        if lbs is None:
            lbs = [-inf]*len(names)
        if ubs is None:
            ubs = [inf]*len(names)
        if vtypes is None:
            vtypes = [gp.GRB.CONTINUOUS]*len(names)
        elif len(vtypes) == 1:
            vtypes = vtypes*len(names)
        return names, vtypes, lbs, ubs

    def get_new_vars(self, names, vtypes=None, lbs=None, ubs=None):
        names, vtypes, lbs, ubs = self.handle_args(names, vtypes, lbs, ubs)
        new_vars = []
        for i in range(len(names)):
            new_vars.append(self.get_new_var(names[i], vtypes[i], lb=lbs[i], ub=ubs[i]))
        #
        if isinstance(names, np.ndarray):
            return np.array(new_vars).reshape(names.shape)
        else:
            return new_vars    

    def assert_constraints(self, constraints):
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
            # TODO: should min be implemented?
            raise NotImplementedError
    
    def assert_matrix_constraint(self, c: MatrixConstraint):
        # TODO: handle 'not equal' constraint (exception)
        x = self.get_new_vars(c.x) # get gurobi vars
        print("x shape is: ", c.x.shape, " b shape is: ", c.b.shape, " A shape is: ", c.A.shape)
        self.matrix_helper(c.A, x, c.type, c.b)
    
    def assert_simple_constraint(self, c: Constraint):
        # Note: if matrix interface is causing problems, 
        # can add constraints by building up gurobi TempConstr one monomial at a time
        # e.g. g1 = gurobi_x + gurobi_y,  g2 = g1 + 2*z ... etc.
        A = np.array([m.coeff for m in c.monomials]).reshape(1,-1)
        b = np.array([c.scalar]) #.reshape(1,1)
        x = np.array(self.get_new_vars([m.var for m in c.monomials])).reshape(-1,1)
        self.matrix_helper(A, x, c.type, b)
    
    def matrix_helper(self, A, x, R, b):
        """
        Assert a gurobi matrix constraint of a certain type R.
        x should be GUROBI variables
        R should be an instance of a ConstraintType
        """
        if R == ConstraintType('EQUALITY'):
            self.model.addConstr(A @ x == b)
        elif R == ConstraintType('LESS_EQ'):
            self.model.addConstr(A @ x <= b)
        elif R == ConstraintType('GREATER_EQ'):
            self.model.addConstr(A @ x >= b)
        else:
            raise NotImplementedError

    def assert_max_constraint(self, c: MaxConstraint):
        varsin = self.get_new_vars([c.var1in, c.var2in])
        varout = self.get_new_vars(c.varout)
        if self.PWL_encoding == 'native':
            self.model.addConstr(varout == max_(varsin))
        elif self.PWL_encoding == 'MIPverify':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def assert_relu_constraint(self, c: ReluConstraint):
        if self.PWL_encoding == 'native':
            max_version = MaxConstraint(varout=c.varout, varsin=[c.varin, 0])
            self.assert_max_constraint(max_version)
        elif self.PWL_encoding == 'MIPverify':
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def check_sat(self, output_filename="", timeout=0, vars_of_interest=[], verbose=True, dnc=True, tried=0):
        """
        Check if constrained problem is feasible / satisfiable. 
        Note: has some args from marabou interface that do nothing, like dnc. 
        will be removed later when these args are moved to solver() init call
        """
        self.model.setObjective(0, gp.GRB.MAXIMIZE)
        try:
            self.model.optimize()
            status = self.model.status
            
            if status == gp.GRB.INFEASIBLE:
                # note, can compute irreducible inconsistent subsystem...
                return Result.UNSAT, {}, {}
            elif status == gp.GRB.OPTIMAL:
                vals = self.model.getVars()
                vals_in_BMC_vars = {v.varName: v.x for v in vals}
                return Result.SAT, vals_in_BMC_vars, {}
            # todo: allow for suboptimal but feasible solutions?
            elif status == gp.GRB.Status.INF_OR_UNBD and tried == 0:
                self.model.setParam(gp.GRB.Param.DualReductions, 0)
                return self.check_sat(output_filename=output_filename, tried=1)
            else:
                print('Optimization was stopped with status %d' % status)
                return Result.ERROR, {}, {}

        except gp.GurobiError as e:
            print("Gurobi Error Caught.")
            return Result.ERROR, {}, {}







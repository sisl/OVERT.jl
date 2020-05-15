from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint, MaxConstraint
import numpy as np
from maraboupy import Marabou, MarabouCore, DnCSolver, DnC
from multiprocessing import Process, Pipe
import os
from MC_interface import Result
import pickle
# solver
class MarabouWrapper():
    """
    A class which converts to the maraboupy interface.
    """
    def __init__(self, n_worker=4):
        # initialize "clean" query
        self.clear()
        self.n_worker = n_worker
        self.eq_type_map = {ConstraintType('EQUALITY'): MarabouCore.Equation.EQ,
                            ConstraintType('LESS_EQ') : MarabouCore.Equation.LE,
                            ConstraintType('GREATER_EQ'): MarabouCore.Equation.GE}

    def clear(self):
        # clear marabou query
        self.ipq = MarabouCore.InputQuery()
        self.ipq.setNumberOfVariables(0)
        self.variable_map = {} # maps string names -> integers
        self.input_vars = []
        self.constraints = [] # log of things that have been asserted, for debug/double check
        self.num_relu = 0

    def assert_constraints(self, constraints):
        # add constraints to Marabou IPQ object
        for c in constraints:
            self.assert_one_constraint(c)

    def assert_one_constraint(self, constraint):
        self.constraints.append(constraint)
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
    
    def assert_matrix_constraint(self, constraint):
        # form: Ax R b
        # get new vars for x variables
        marabou_vars = self.get_new_vars(constraint.x.flatten())
        # for every row in A, add a Marabou 'equation'
        for row in range(constraint.A.shape[0]):
            coefficients = constraint.A[row, :]
            scalar = constraint.b[row]
            self.add_marabou_eq(coefficients, marabou_vars, constraint.type, scalar)
    
    def assert_relu_constraint(self, relu):
        if len(np.array(relu.varin).flatten()) > 1:
            print("ERROR: relu.varin is not scalar! It has length", len(relu.varin))
            raise NotImplementedError
        else: # truly, the ok case
            self.num_relu += 1
            MarabouCore.addReluConstraint(self.ipq, self.get_new_var(relu.varin), self.get_new_var(relu.varout))
            #MarabouCore.addReluConstraint(self.ipq, self.get_new_var(relu.varin), self.get_new_var(relu.varout), self.num_relu)
    
    def assert_max_constraint(self, c):
        MarabouCore.addMaxConstraint(self.ipq, {self.get_new_var(c.var1in), self.get_new_var(c.var2in)}, self.get_new_var(c.varout))
    
    def add_marabou_eq(self, coeffs, variables, eq_type, scalar):
        if eq_type in [ConstraintType('LESS'), ConstraintType('GREATER')]:
            raise NotImplementedError
            # TODO: apply epsilon conversion by adding a slack variable = epsilon
            # to convert from a strict inequality to a non-strict one
        elif eq_type == ConstraintType('NOT_EQUAL'):
            raise NotImplementedError
        assert(len(coeffs) == len(variables))
        eq = MarabouCore.Equation(self.eq_type_map[eq_type])
        for i in range(len(coeffs)):
            eq.addAddend(coeffs[i], variables[i])
        eq.setScalar(scalar)
        self.ipq.addEquation(eq)

    def get_new_vars(self, MC_vars):
        return [self.get_new_var(v) for v in MC_vars]
    
    def get_new_var(self, v):
        """
        Returns a new marabou variable IFF one does not already exist for this
        MC variable string name.
        """
        if not (v in self.variable_map.keys()):
            self.variable_map[v] = self.ipq.getNumberOfVariables()
            self.ipq.setNumberOfVariables(self.ipq.getNumberOfVariables() + 1)
        return self.variable_map[v]

    def assert_init(self, init_set):
        """ assert states in InitSet set
        for now, assume set is a box set over inputs
        of the form: {"x@0": (0, 5), "theta@0": (-np.pi/4, np.pi/4)}
        """
        for k in init_set.keys():
            input_var = self.get_new_var(k)
            self.input_vars.append(input_var)
            lower_bound = init_set[k][0]
            upper_bound = init_set[k][1] 
            self.ipq.setLowerBound(input_var, lower_bound)
            self.ipq.setUpperBound(input_var, upper_bound)
        for i in range(len(self.input_vars)):
            self.ipq.markInputVariable(self.input_vars[i], i)
    
    def get_bounds(self):
        lb = {}
        ub = {}
        for v in range(self.ipq.getNumberOfVariables()):
            lb[v] = self.ipq.getLowerBound(v)
            ub[v] = self.ipq.getUpperBound(v)

        return lb, ub

    def setup_SBT(self):
        # set up the symbolic bound tightener
        pass

    # inspired by MarabouNetwork.py::solve in Marabou/maraboupy
    def check_sat(self, output_filename="", timeout=0, vars_of_interest=[], verbose=True, dnc=True):
        # todo: redirect output to cwd/maraboulogs/
        if (not dnc) or (self.n_worker == 1):
            options = Marabou.createOptions(timeoutInSeconds=timeout)
        else: # dnc
            options = Marabou.createOptions(timeoutInSeconds=timeout, dnc=True, verbosity=0,
                                            initialDivides=2, initialTimeout=120, numWorkers=self.n_worker)
            # options = Marabou.createOptions(timeoutInSeconds=timeout, dnc=True, verbosity=0,
            #                                 initialDivides=2, initialTimeout=120, numWorkers=self.n_worker,
            #                                 biasStrategy="estimate", focusLayer=1000, lookAheadPreprocessing=True)
        MarabouCore.saveQuery(self.ipq, "query_dump")
        vals, stats = MarabouCore.solve(self.ipq, options, output_filename)
        self.convert_sat_vals_to_mc_vars(vals)
        if verbose:
            self.print_results(vals, stats, vars_of_interest=vars_of_interest)
        if stats.hasTimedOut():
            return Result.TIMEOUT, self.vals_with_mc_vars, stats
        elif len(vals) == 0:
            return Result.UNSAT, self.vals_with_mc_vars, stats
        else: # len(vals) /== 0
            return Result.SAT, self.vals_with_mc_vars, stats
    
    # directly inspired by MarabouNetwork.py::solve in Marabou/maraboupy
    def print_results(self, vals, stats, vars_of_interest=[]):
        if stats.hasTimedOut():
                print("TO")
        elif len(vals)==0:
            print("UNSAT")
        else:
            print("SAT")
            #
            print("values: ", self.vals_with_mc_vars)
            # for i in range(len(self.input_vars)):
            #     print("input ", self.input_vars[i], " = ", vals[self.inputVars[i]])
            # for i in range(self.outputVars.size):
            #     print("output {} = {}".format(i, vals[self.outputVars.item(i)]))
            # TODO: add printing for output vars and some subset of vars you care about (vars_of_interest)

    def convert_sat_vals_to_mc_vars(self, vals):
        inverted_var_map = {value: key for key, value in self.variable_map.items()}
        self.vals_with_mc_vars = {(inverted_var_map[k],v) for k,v in vals.items()}
        
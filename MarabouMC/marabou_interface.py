from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint
import numpy as np
from maraboupy import MarabouCore
from MC_interface import Result

# solver
class MarabouWrapper():
    """
    A class which converts to the maraboupy interface.
    """
    def __init__(self):
        # initialize "clean" query
        self.clear()
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
        else:
            raise NotImplementedError
    
    def assert_simple_constraint(self, constraint):
        eq = MarabouCore.Equation(self.eq_type_map[constraint.type])
        for m in constraint.monomials:
            eq.addAddend(m.coeff, self.get_new_var(m.var))
        eq.setScalar(constraint.scalar)
        self.ipq.addEquation(eq)
    
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
        MarabouCore.addReluConstraint(self.ipq, self.get_new_var(relu.varin), self.get_new_var(relu.varout))
    
    def add_marabou_eq(self, coeffs, variables, eq_type, scalar):
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
    def check_sat(self, output_filename="", timeout=0, vars_of_interest=[], verbose=True):
        # todo: redirect output to cwd/maraboulogs/
        vals, stats = MarabouCore.solve(self.ipq, output_filename, timeout)
        if verbose:
            self.print_results(vals, stats, vars_of_interest=vars_of_interest)
        if stats.hasTimedOut():
            return Result.TIMEOUT, vals, stats
        elif len(vals) == 0:
            return Result.UNSAT, vals, stats
        else: # len(vals) /== 0
            return Result.SAT, vals, stats
    
    # directly inspired by MarabouNetwork.py::solve in Marabou/maraboupy
    def print_results(self, vals, stats, vars_of_interest=[]):
        if stats.hasTimedOut():
                print("TO")
        elif len(vals)==0:
            print("UNSAT")
        else:
            print("SAT")
            #
            inverted_var_map = {value: key for key, value in self.variable_map.items()}
            print("values: ", {(inverted_var_map[k],v) for k,v in vals.items()})
            # for i in range(len(self.input_vars)):
            #     print("input ", self.input_vars[i], " = ", vals[self.inputVars[i]])
            # for i in range(self.outputVars.size):
            #     print("output {} = {}".format(i, vals[self.outputVars.item(i)]))
            # TODO: add printing for output vars and some subset of vars you care about (vars_of_interest)

        
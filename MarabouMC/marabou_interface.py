from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint
import numpy as np
from maraboupy import MarabouCore

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
            self.add_marabou_eq(coefficients, marabou_vars, scalar, constraint.type)
    
    def add_marabou_eq(coeffs, variables, scalar, eq_type):
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
            self.variable_map[v] = self.ipq.getNumberOfVariables() + 1
            self.ipq.setNumberOfVariables(self.ipq.getNumberOfVariables() + 1)
        return self.variable_map[v]

    def assert_init(self, set, states):
        # assert states /in InitSet set
        # also used for marking inputs to marabou
        # TODO: how flexibly should set be represented?
        pass

    def convert(self, constraints):
        """
        Takes in a set of constraints and converts them to a format that Maraboupy understands.
        # something like "getMarabouQuery" in MarabouNetwork.py in maraboupy
        """
        pass

    def setup_SBT(self):
        # set up the symbolic bound tightener
        pass

    def check_sat(self):
        # call convert to convert internal representation of timed contraints to marabou vars + ineqs
        # something like "solve" in MarabouNetwork.py in maraboupy
        pass
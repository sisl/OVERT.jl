from MC_constraints import Constraint, MatrixConstraint, MaxConstraint, ReluConstraint
import numpy as np
class OverapproxVerifier:
    """
    A function to verify the validity of an overapproximation using dreal. 
    Translates Constraint objects into smtlib2
    """
    def __init__(self, phi, phihat):
        self.phi = phi # the true function
        self.phihat = phihat # the overapproximation
        self.smtlib_formula = [] # the formula not(phi => phihat) in smtlib2
        self.formula_converter = FormulaConverter() 

    def check(self):
        self.convert_formula()
        self.run_dreal()

    def convert_formula(self):
        """
        Converts to smtlib2 (possibly write to file?)
        """
        pass

    def run_dreal(self):
        """
        Runs dreal on the formula in smtlib2 format. 
        """
        pass

class FormulaConverter:
    """
    A class to convert MarabouModelChecker Constraint type into smtlib2.
    Meta-programming.
    """
    def __init__(self):
        self.var_count = 0
        self.var_map = {}

    def prefix_notate(self, op, args):
        """
        Turn an op into its prefix form: (op arg1 arg2 ...)
        """
        expr = "(" + op + " "
        for elem in args:
            expr += str(elem) + " "
        expr = expr[0:-1]
        expr += ")"
        return expr
    
    def declare_const(self, constname, consttype):
        return self.prefix_notate("declare-const", [constname, consttype])
    
    def define_atom(self, atomname, atomvalue):
        eq_expr = self.prefix_notate("=", [atomname, atomvalue])
        return self.assert_statement(eq_expr)
    
    def assert_statement(self, expr):
        return self.prefix_notate("assert", [expr])
    
    def negate(self, expr):
        return self.prefix_notate("not", [expr])
    
    def footer(self, expr):
        return ["(check-sat)", "(get-model)"]
    
    def header(self, expr):
        """
        define max and relu functions.
        """
        return [self.define_max(), self.define_relu()]

    def define_max(self):
        return "(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))"
    
    def define_relu(self):
        return "(define-fun relu ((x Real)) Real (max x 0))"
    
    def assert_conjunction(self, constraint_list, conjunct_name=None):
        formula, conjunct_name = self.declare_conjunction(constraint_list, conjunct_name=conjunct_name) # declare conjunction
        formula += [self.assert_statement(conjunct_name)] # assert conjunction
        return formula

    def assert_negated_conjunct(self, constraint_list):
        pass

    def declare_conjunction(self, constraint_list, conjunct_name=None):
        """
        Given a list of constraints, declare their conjunction but DO NOT
        assert their conjunction.
        e.g. 
        (declare ... A)
        (assert A = (<= y 5))
        (declare ... B)
        (assert B = (>= z 6))
        ...
        (declare ... phi)
        (assert (= phi (and A B)))
        But notice we are just _defining_ phi, we are not asserting that
        phi _holds_, which would be: (assert phi) [not doing that tho!]
        """
        bool_var_defs, bool_var_names = self.declare_list(constraint_list)
        if conjunct_name is None:
            conjunct_name = self.get_new_var()
        conjunct = self.prefix_notate("and", bool_var_names)
        formula = bool_var_defs + [self.define_atom(conjunct_name, conjunct)]
        return formula, conjunct_name

    def declare_list(self, constraint_list):
        """
        turn a list of some type of AbstractConstraint <: into 
        smtlib declarations/definitions:
        (declare ... A)
        (assert A = (< y 5))
        But DON'T assert either true or false for the atom A
        """
        bool_var_defs = [] # definitions + declarations
        bool_var_names = [] # names
        for item in constraint_list:  
            if isinstance(item, Constraint):
                expr = self.convert_Constraint(item)
            elif isinstance(item, MatrixConstraint):
                expr = self.convert_MatrixConstraint(item)
            elif isinstance(item, ReluConstraint):
                expr = self.convert_ReluConstraint(item)
            elif isinstance(item, MaxConstraint):
                expr = self.convert_MaxConstraint(item)
            else:
                raise NotImplementedError
            for e in expr:
                bool_var = self.get_new_var()
                bool_var_names += [bool_var]
                bool_var_defs += [self.declare_const(bool_var, "Bool")]
                bool_var_defs += [self.define_atom(bool_var, e)]
        return bool_var_defs, bool_var_names

    def convert_Constraint(self, c):
        """
        Convert a constraint of type Constraint:
        self.type 
        self.monomials
        self.scalar
        to: 
        5*x - 6*y <= 7
        (<= (+ (* 5 y) (* -6 y)) 7)
        # + can take any number of args BUT unary minus must be handled separately: (- 5)
        """
        coeffs = [m.coeff for m in c.monomials]
        variables = [m.var for m in c.monomials]
        return self.convert_constraint_helper(coeffs, variables, c.type.__repr__(), c.scalar)
    
    def convert_constraint_helper(self, coeffs, variables, eq_type, scalar):
        # build up list of monomials: (* 5 y), (* (- 6) x), applying unary minus where necessary
        m_list = []
        for i in range(len(variables)):
            if coeffs[i] < 0:
                # apply unary minus
                smtlib_coeff = self.prefix_notate("-", [abs(coeffs[i])])
            else:
                smtlib_coeff = coeffs[i]
            m_list += [self.prefix_notate("*", [smtlib_coeff, variables[i]])]
        # add monomials together
        left_side = self.prefix_notate("+", m_list)
        # construct right side
        if scalar < 0:
            right_side = self.prefix_notate("-", [abs(scalar)])
        else:
            right_side = scalar
        # construct (R (monomials) scalar)
        return [self.prefix_notate(eq_type, [left_side, right_side])]
        
    def convert_MatrixConstraint(self, c: MatrixConstraint):
        """
        Convert a constraint of type MatrixConstraint.
        Ax = b 
        to a list of 1D constraints in smtlib
        """
        constraints = []
        m = c.A.shape[0]
        for row in range(m):
            constraints += self.convert_constraint_helper(c.A[row,:], c.x.flatten(), c.type.__repr__(), c.b[row][0])
        return constraints

    def convert_ReluConstraint(self, c):
        """
        Convert a constraint of type ReluConstraint
        varin # may be of length > 1 # todo handle multi D
        varout
        to:
        (= (relu x) y)
        """
        varsin = np.array(c.varin, dtype='object').flatten() # allows handling multi-dimensional inputs
        varsout = np.array(c.varout, dtype='object').flatten()
        l = []
        for i in range(len(varsin)):
            left_side = self.prefix_notate("relu", [varsin[i]])
            right_side = varsout[i]
            l += [self.prefix_notate("=", [left_side, right_side])]
        return l
        
    def convert_MaxConstraint(self, c):
        """
        Convert a constraint of type MaxConstraint
        """
        # does NOT handle multi dimensional inputs
        assert(not isinstance(c.var1in, np.ndarray) and not isinstance(c.var1in, list))
        left_side = self.prefix_notate("max", [c.var1in, c.var2in])
        right_side = c.varout
        return [self.prefix_notate("=", [left_side, right_side])]

    def get_new_var(self):
        self.var_count += 1
        return "b"+str(self.var_count) # b for boolean


    


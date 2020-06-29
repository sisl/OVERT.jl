from MC_constraints import Constraint, MatrixConstraint, MaxConstraint, ReluConstraint, Monomial, NLConstraint
import numpy as np
import os
from MC_interface import Result

class SMTLibFormula:
    def __init__(self, formula):
        self.formula = formula
    
    def __repr__(self): 
        return "\n".join(self.formula)

    def _print(self, dirname="", fname=""):
        """ 
        Print to stdout or to file
        """
        if fname == "":
            print(self.__repr__())
        else:
            if dirname == "":
                # put in subdir of cwd
                dirname = os.path.join(os.getcwd(), "smtlib_files")
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
            abs_fname = os.path.join(dirname, fname)
            fhandle = open(abs_fname+".smtlib2", 'wt')
            fhandle.write(self.__repr__())

class OverapproxVerifier:
    """
    A function to verify the validity of an overapproximation using dreal. 
    Translates Constraint objects into smtlib2
    """
    def __init__(self, phi, phihat):
        assert(isinstance(phi, list))
        assert(isinstance(phihat, list))
        self.phi = phi # the true function
        self.phihat = phihat # the overapproximation
        self.smtlib_formula = [] # the formula not(phi => phihat) aka (phi and not phihat) in smtlib2
        self.f = FormulaConverter() 

    def check(self):
        self.create_smtlib_script()
        self.run_dreal()

    def convert_formula(self):
        """
        Converts to smtlib2 (possibly write to file?)
        Assert the formula not(phi => phihat) aka (phi and not phihat)
        """
        self.smtlib_formula = ["; assert phi"]
        phi = self.f.assert_logical(self.phi)
        self.smtlib_formula += phi
        # 
        self.smtlib_formula += ["; assert not phi-hat"]
        notphihat = self.f.assert_negated_logical(self.phihat)
        self.smtlib_formula += notphihat # combine constraint lists
        print(self.f) #  for debug, mainly
        
    def create_smtlib_script(self):
        self.convert_formula() # populates smtlib_formula with details of formula
        self.smtlib_formula = self.f.header() + \
                              self.f.declare_reals() + \
                              self.smtlib_formula + \
                              self.f.footer()
        self.formula_object = SMTLibFormula(self.smtlib_formula)
        self.formula_object._print()

    def run_dreal(self):
        """
        Runs dreal on the formula in smtlib2 format. 
        """
        pass

class StatefulDrealWrapper:
    """
    A wrapper like the one in marabou_interface.py
    that collects constraints as they are asserted.
    Should implement a check_sat() method with calls dreal.
    Will contain a FormulaConverter object and use
    similar logic to OverApproxVerifier class. 
    """
    def __init__(self):
        self.clear() 

    def clear(self):
        """
        Clear things to create a new query
        """
        self.f = FormulaConverter()
        self.constraints = []
        self.smtlib_formula = []
        self.input_vars = []
    
    def convert(self):
        """
        Convert constraint objects to SMTLIB2
        """
        # TODO: I assert_logical of constraints after "declaring reals"
        # so there are no reals to declare yet and no reald are then declared. 
        # I should either:
        # declare reals inside assert_logical OR parse the constraints before
        # putting things inside the smtlib_formula
        main_formula = self.f.assert_logical(self.constraints)
        self.smtlib_formula = self.f.header() + \
                              self.f.declare_reals() + \
                              main_formula + \
                              self.f.footer()
        self.formula_object = SMTLibFormula(self.smtlib_formula)

    def assert_init(self, init_set):
        """
        assert that states are in init set
        """
        for k in init_set.keys():
            self.input_vars.append(k)
            LB = init_set[k][0]
            UB = init_set[k][1]
            cLB = Constraint('GREATER_EQ', monomials=[Monomial(1., k)], scalar=LB)
            cUB = Constraint('LESS_EQ', monomials=[Monomial(1., k)], scalar=UB)
            self.constraints += [cLB, cUB]

    def assert_constraints(self, constraints):
        """
        Assert multiple constraints.
        """
        self.constraints += constraints
    
    def mark_outputs(self, outputs):
        pass

    def check_sat(self):
        """
        Check whether query is satisfiable.
        """
        self.convert()
        # and then call dreal!
        self.formula_object._print()
        fake_values = dict(zip(self.input_vars, np.zeros(len(self.input_vars)))) # TODO: CHANGE
        fake_stats = []
        return Result.UNSAT, fake_values, fake_stats

class FormulaConverter:
    """
    A class to convert MarabouModelChecker Constraint type into smtlib2.
    Meta-programming.
    The functions in this class are (mostly) intended to be pure; i.e.
    given some arguments they return the result, and there are no
    side - effects.
    """
    def __init__(self):
        self.new_var_count = 0
        self.var_list = {'bools':[], 
                        'reals':[]}

    def __repr__(self):
        s = ""
        s += "~~~~~~~~~~Formula Converter~~~~~~~~~~\n"
        s += "new_var_count = " + str(self.new_var_count) + " "
        s += "bools: " + " ".join(self.var_list['bools']) + " "
        s += "reals: " + " ".join(self.var_list['reals'])
        s += "\n~~~~~~~~~~\n"
        return s
    
    def assert_logical(self, f): # changed to: assert_conjunction in julia
        """
        Assert expression. f is a list.
        Returns a list.
        """
        if len(f) == 1:
            return [self.assert_statement(self.convert_any_constraint(f[0])[0])]
        elif len(f) > 1:
            # assert conjunction
            return self.assert_conjunction(f)
        else: # empty list
            return []
    
    def assert_negated_logical(self, f): # changed to: assert_negated_conjunction in julia
        """
        Assert the negation of an expression. f is a list.
        Returns a list.
        """
        if len(f) == 0:
            return []
        elif len(f) == 1:
            return [self.assert_statement(self.negate(self.convert_any_constraint(f[0])[0]))]
        else: # len(f) > 1
            return self.assert_negated_conjunction(f)
        
    def add_real_var(self, v):
        if v not in self.var_list['reals']:
            self.var_list['reals'].append(v)
    
    def add_real_vars(self, vlist):
        for v in vlist:
            self.add_real_var(v)
        
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
    
    def declare_reals(self):
        real_decls = []
        for v in self.var_list['reals']:
            real_decls += [self.declare_const(v, 'Real')]
        return real_decls

    def define_atom(self, atomname, atomvalue):
        eq_expr = self.prefix_notate("=", [atomname, atomvalue])
        return self.assert_statement(eq_expr)
    
    def assert_statement(self, expr):
        return self.prefix_notate("assert", [expr])
    
    def negate(self, expr):
        return self.prefix_notate("not", [expr])
    
    def footer(self):
        return ["(check-sat)", "(get-model)"]
    
    def header(self):
        """
        define max and relu functions.
        """
        h = [self.set_logic(), self.produce_models()]
        h += [self.define_max(), self.define_relu()]
        return h

    def set_logic(self):
        return "(set-logic ALL)"
    
    def produce_models(self):
        return "(set-option :produce-models true)"

    def define_max(self):
        return "(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))"
    
    def define_relu(self):
        return "(define-fun relu ((x Real)) Real (max x 0))"
    
    def assert_conjunction(self, constraint_list, conjunct_name=None):
        formula, conjunct_name = self.declare_conjunction(constraint_list, conjunct_name=conjunct_name) # declare conjunction
        formula += [self.assert_statement(conjunct_name)] # assert conjunction
        return formula

    def assert_negated_conjunction(self, constraint_list, conjunct_name=None):
        """
        Assert the negation of conjunction of the constraints passed in constraint_list.
        not (A and B and C and ...)
        """
        formula, conjunct_name = self.declare_conjunction(constraint_list, conjunct_name=conjunct_name) # declare conjunction
        formula += [self.assert_statement(self.negate(conjunct_name))] # assert NEGATED conjunction
        return formula

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
            conjunct_name = self.get_new_bool()
        assert(len(bool_var_names) > 1)
        conjunct = self.prefix_notate("and", bool_var_names)
        conjunct_decl = [self.declare_const(conjunct_name, "Bool")]
        conjunct_def = [self.define_atom(conjunct_name, conjunct)]
        formula = bool_var_defs + conjunct_decl + conjunct_def
        return formula, conjunct_name
    
    def assert_disjunction(self, constraint_list, disjunct_name=None):
        raise NotImplementedError

    def convert_any_constraint(self, item):
        if isinstance(item, Constraint):
            expr = self.convert_Constraint(item)
        elif isinstance(item, MatrixConstraint):
            expr = self.convert_MatrixConstraint(item)
        elif isinstance(item, ReluConstraint):
            expr = self.convert_ReluConstraint(item)
        elif isinstance(item, MaxConstraint):
            expr = self.convert_MaxConstraint(item)
        elif isinstance(item, NLConstraint):
            expr = self.convert_NLConstraint(item)
        else:
            raise NotImplementedError
        return expr

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
            expr = self.convert_any_constraint(item)
            for e in expr:
                bool_var = self.get_new_bool()
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
        self.add_real_vars(variables) # log in var map
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
        c.b = c.b.reshape(-1,1)
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
        self.add_real_vars(np.hstack((varsin, varsout)).flatten())  # log in var map
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
        self.add_real_vars([c.var1in, c.var2in]) # log in var map
        left_side = self.prefix_notate("max", [c.var1in, c.var2in])
        right_side = c.varout
        return [self.prefix_notate("=", [left_side, right_side])]
    
    def convert_NLConstraint(self, c):
        """
        Convert a constraint of type NLConstraint
        For now, they are only 1D
        y = sin(x)  ->   (= y (sin x))
        """
        assert isinstance(c.indep_var, str) # not an array of variables
        self.add_real_vars([c.out, c.indep_var])
        left_side = c.out  # aka y
        right_side = self.prefix_notate(c.fun, c.indep_var)
        return [self.prefix_notate(c.type.__repr__(), [left_side, right_side])]

    # TODO: implement mul  and div nonlinear constraint classes

    def get_new_bool(self):
        self.new_var_count += 1
        v = "b"+str(self.new_var_count) # b for boolean
        assert(v not in self.var_list['bools'])
        self.var_list['bools'].append(v)
        return v


    


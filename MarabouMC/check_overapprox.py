from MC_constraints import Constraint, MatrixConstraint, MaxConstraint, ReluConstraint

class OverapproxVerifier:
    """
    A function to verify the validity of an overapproximation using dreal. 
    Translates Constraint objects into smtlib2
    """
    def __init__(self, phi, phihat):
        self.phi = phi # the true function
        self.phihat = phihat # the overapproximation
        self.smtlib = [] # the formula in smtlib2 

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
        return self.prefix_notate("assert", [eq_expr])
    
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
    
    def assert_conjunction(self, constraint_list):
        # conjunct = []
        # for item in constraint_list:
        #     # convert somehow
        #     # then
        #     pass
        # return conjunct
        pass

    def assert_negated_conjunct(self, constraint_list):
        pass

    def declare_list(self, constraint_list):
        """
        turn a list of some type of AbstractConstraint <: into 
        smtlib declarations/definitions:
        (declare ... A)
        (assert A = (< y 5))
        But DON'T assert either true or false for the atom A
        """
        statements = []
        for item in constraint_list:
            bool_var = self.get_new_var()
            statements += [self.declare_const(bool_var, "Bool")]
            if isinstance(item, Constraint):
                expr = self.convert_Constraint(item)
            elif isinstance(item, MatrixConstraint):
                expr = self.convert_MatrixConstraint(item)
            elif isinstance(item, MaxConstraint):
                expr = self.convert_MaxConstraint(item)
            elif isinstance(item, ReluConstraint):
                expr = self.convert_ReluConstraint(item)
            else:
                raise NotImplementedError
            statements += [self.define_atom(bool_var, expr)]
        return statements

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
        # build up list of monomials: (* 5 y), (* (- 6) x), applying unary minus where necessary
        m_list = []
        for m in c.monomials:
            if m.coeff < 0:
                # apply unary minus
                smtlib_coeff = self.prefix_notate("-", [abs(m.coeff)])
            else:
                smtlib_coeff = m.coeff
            m_list += self.prefix_notate("*", [smtlib_coeff, m.var])
        # add monomials together
        left_side = self.prefix_notate("+", m_list)
        # construct right side
        if c.scalar < 0:
            right_side = self.prefix_notate("-", [abs(c.scalar)])
        else:
            right_side = c.scalar
        # construct (R (monomials) scalar)
        return self.prefix_notate(c.type.__repr__(), [left_side, right_side])
        
    def convert_MatrixConstraint(self):
        """
        Convert a constraint of type MatrixConstraint
        """
        pass

    def convert_ReluConstraint(self, c):
        """
        Convert a constraint of type ReluConstraint
        varin # may be of length > 1 # todo handle multi D
        varout
        to:
        (= (relu x) y)
        """
        try: # handle multi dimensional inputs
            l = []
            for i in range(len(c.varin)):
                left_side = self.prefix_notate("relu", [c.varin[i]])
                right_side = c.varout[i]
                l += [self.prefix_notate("=", [left_side, right_side])]
            return l
        except: # single dimensional inputs
            left_side = self.prefix_notate("relu", c.varin)
            right_side = c.varout
            return self.prefix_notate("=", [left_side, right_side])
        
    def convert_MaxConstraint(self, c):
        """
        Convert a constraint of type MaxConstraint
        """
        try: # handle multi dimensional inputs
            l = []
            for i in range(len(c.varin)):
                left_side = self.prefix_notate("max", [c.var1in[i], c.var2in[i]])
                right_side = c.varout[i]
                l += [self.prefix_notate("=", [left_side, right_side])]
            return l
        except: # single dimensional inputs
            left_side = self.prefix_notate("max", [c.var1in, c.var2in])
            right_side = c.varout
            return self.prefix_notate("=", [left_side, right_side])

    def get_new_var(self):
        self.var_count += 1
        return "b"+str(self.var_count) # b for boolean


    


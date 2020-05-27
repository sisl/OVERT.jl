from enum import Enum
#import tensorflow as tf
# from maraboupy import *
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint, MaxConstraint
from Constraint_utils import matrix_equality_constraint, equality_constraint
from properties import Property, ConstraintProperty
from transition_systems import TransitionRelation
from copy import deepcopy
import numpy as np

class Result(Enum):
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3
    TIMEOUT = 4
    
# inspired by cosa2 model checker
def substitute_c(c: Constraint, mapping): 
    """
    substitute(x+y<0, [x => x@1, y => y@1])
    Used in unroller.
    Return a NEW object. Do not modify input arg.
    """
    new_c = deepcopy(c)
    for mono in new_c.monomials:
        mono.var = mapping[mono.var]
    return new_c

def substitute_Mc(c: MatrixConstraint, mapping): 
    """
    substitute(x+y<0, [x => x@1, y => y@1])
    Used in unroller.
    Return a NEW object. Do not modify input arg.
    """
    new_c = deepcopy(c)
    xshape = new_c.x.shape
    new_c.x = np.array([mapping[v] for v in new_c.x.flatten()]).reshape(xshape)
    return new_c

def substitute_relu(c: ReluConstraint, mapping):
    """
    substitute(y = relu(x), [x => x@1, y => y@1])
    Used in unroller.
    Return a NEW object. Do not modify input arg.
    """
    new_c = deepcopy(c)
    new_c.varin = mapping[c.varin]
    new_c.varout = mapping[c.varout]
    return new_c

def substitute_max(c: MaxConstraint, mapping):
    """
    substitute(y = max(w, v), [w => w@1, v => v@1, y => y@1])
    Used in unroller.
    Return a NEW object. Do not modify input arg.
    """
    new_c = deepcopy(c)
    new_c.var1in = mapping[c.var1in]
    new_c.var2in = mapping[c.var2in]
    new_c.varout = mapping[c.varout]
    return new_c

def timer_helper(var_list, t):
    return [v+"@"+str(t) if not isprimed(v) else v[:-1]+"@"+str(t+1) for v in var_list]

def isprimed(var):
    # if variable has ' in namestring, it is "primed"
    return ("'" in var)

# inspired by cosa2 model checker
class Unroller():
    def __init__(self):
        pass
        self.cache = None # dictionary from original to timed version (may be helpful)

    def at_time_constraint(self, c, t):
        """
        at_time(x+y'<0, 2)
            return x@2 + y@3 < 0
        The timestep passed corresponds to the _unprimed_ variable.
        c is a Constraint or MatrixConstraint
        """
        if isinstance(c, MatrixConstraint):
            vars_in_c = c.x.flatten()
            timed_vars = timer_helper(vars_in_c, t)
            return substitute_Mc(c, dict(zip(vars_in_c, timed_vars)))
        elif isinstance(c, Constraint):
            vars_in_c = [m.var for m in c.monomials]
            timed_vars = timer_helper(vars_in_c, t)
            return substitute_c(c, dict(zip(vars_in_c, timed_vars)))
        elif isinstance(c, ReluConstraint):
            vars_in_c = np.array([c.varin, c.varout]).flatten()
            timed_vars = timer_helper(vars_in_c, t)
            return substitute_relu(c, dict(zip(vars_in_c, timed_vars)))
        elif isinstance(c, MaxConstraint):
            vars_in_c = np.array([c.var1in, c.var2in, c.varout]).flatten()
            timed_vars = timer_helper(vars_in_c, t)
            return substitute_max(c, dict(zip(vars_in_c, timed_vars)))
        else:
            raise NotImplementedError

    def at_time_relation(self, TR: TransitionRelation, tstep):
        # for every constraint in the transition relation, call at_time and do subsitution
        timed_constraints = []
        for c in TR.constraints: 
            timed_constraints.append(self.at_time_constraint(c, tstep))
        return timed_constraints
    
    def at_time_property(self, prop, tstep):
        # ALWAYS ASSERT THE COMPLEMENT OF THE PROPERTY!!!
        timed = []
        for ccomp in prop.constraint_complements:
            timed.append(self.at_time_constraint(ccomp, tstep))
        return timed

    def at_time_init(self, init_set):
        ## for now, assume boxes. Single dict.
        timed_init_set = {}
        for k in init_set.keys():
            timed_init_set[k+"@0"] = init_set[k]
        return timed_init_set

    def untime(self):
        """
        From timed constraint to x, x' constraint
        """
        pass
        # implement later


# model checker functions
"""
Should be able to run a model checking algo on a specific transition system 
tuple <X, I(x), T(x,x')> and a property
"""

# inspired by cosa2 model checker
# BMC , a model checking algo
class BMC():
    def __init__(self, ts, prop_file = "", prop=None, solver=None):
        self.transition_sys = ts 
        if prop:
            self.prop = prop
        else:
            self.prop = Property(prop_file) 
        self.solver = solver
        self.unroller = Unroller()

    def load_prop(self):
        """
        Load prop from file
        """
        pass

    def step_invariant(self, t) -> bool:
        """
        Check that property p holds at time t. Does not assume holds at time 1:t-1
        """
        self.solver.clear() # new query
        # assert that state begins in init set
        self.solver.assert_init(self.unroller.at_time_init(self.transition_sys.initial_set))

        # unroll transition relation for time 0 to t
        for j in range(t):
            self.solver.assert_constraints(self.unroller.at_time_relation(self.transition_sys.transition_relation, j))
        
        # assert complement(property) at time t
        self.prop.complement()
        self.solver.mark_outputs(timer_helper(self.prop.outputs, t)) # mark the outputs 
        self.solver.assert_constraints(self.unroller.at_time_property(self.prop, t))

        return self.solver.check_sat() # result, values, stats

    def check_invariant_until(self, time):
        """
        Check invariant property p holds until time t
        """
        # For now "incremental" sort of. 
        values, stats = None, None
        for i in range(time):
            result, values, stats = self.step_invariant(i) # checks that property holds at i
            if not (result == Result.UNSAT):
                print("Property may not hold at time ", i)
                return result, values, stats
        return Result.UNSAT, values, stats ## TODO: make sure this is correct
    
    def step_invariant_assume(self, t):
        """
        Check that property p holds at time t. Assumes it holds at 1:t-1, for performance
        """
        pass

    def check_invariant_until_assume(self, time):
        """
        Check invariant property p holds until time t, by checking it holds for 1:2, 1:3, ... 1:t
        """
        # For now "incremental" sort of. 
        for i in range(time):
            if not self.step_invariant_assume(i): # checks that property holds at i
                print("Property does not hold at time ", i)
                return Result.SAT 
        return Result.UNSAT ## TODO: make sure this is correct

def CEGAR(algo, max_iters, epsilon):
    """
    (Counter Example Guided) Abstraction Refinement.
    """    
    # while property has not been verified and the maximum number of iterations
    # has not yet been exceeded, refine and try again
    #       algo.transition_system.abstract(epsilon, CEx)
    # if property does not hold, return CEx
    # else if it does hold, return :HOLDS
    pass


    
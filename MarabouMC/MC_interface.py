from enum import Enum
#import tensorflow as tf
# from maraboupy import *
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint
from MC_TF_parser import TFConstraint
from Constraint_utils import matrix_equality_constraint, equality_constraint
from copy import deepcopy
import numpy as np

class Result(Enum):
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3
    TIMEOUT = 4
    
class TransitionRelation:
    def __init__(self):
        states = []
        next_states = []
        constraints = [] # list of Constraints

class Controller:
    def __init__(self):
        self.constraints = []

class TFController(Controller):
    def __init__(self, tf_sess=None, network_file="", inputNames=[], outputName=""):
        super().__init__()
        """
        load constraints by parsing network file
        """
        self.tfconstraintobj = TFConstraint(filename=network_file, sess=tf_sess, inputNames=inputNames, outputName=outputName)
        self.constraints = self.tfconstraintobj.constraints
        self.state_inputs = np.array(self.tfconstraintobj.inputVars).flatten().reshape(-1,1)
        self.control_outputs = self.tfconstraintobj.outputVars
        self.relus = self.tfconstraintobj.relus

class Dynamics:
    def __init__(self, fun, states, controls):
        self.fun = fun # python function representing the dynamics
        self.states = states
        self.control_inputs = controls
        self.next_states = np.array([x+"'" for x in states]).reshape(self.states.shape) # [x,y,z] -> [x', y', z']
        self.constraints = [] # constraints over the states and next states

class OVERTDynamics(Dynamics):
    def __init__(self, fun, states, controls):
        super().__init__(fun, states, controls)
        self.abstract_constraints = [] # constraints over the states and next states, linearized

    def abstract(self, initial_set, epsilon=1e-1):
        """
        Convert dynamics constraints from nonlinear to linearized using OVERT and epsilon.
        """
        pass
        # fill self.abstract_constraints
        # add constraints matching self.next_states and what comes out of abstraction generator

class ControlledTranstionRelation(TransitionRelation):
    def __init__(self, controller_file=None, dynamics_obj=None, epsilon=1-6):
        super().__init__() # states, next states, constraints containers
        self.controller_file = controller_file
        self.dynamics_obj = dynamics_obj
        self.epsilon = epsilon

class TFControlledOVERTTransitionRelation(ControlledTranstionRelation):
    
    def __init__(self, dynamics_obj, controller_file="", controller_obj=None, epsilon=1e-6):
        """
        Constructor takes objs XOR filenames.
        """
        super().__init__(controller_file=controller_file, dynamics_obj=dynamics_obj, epsilon=epsilon)
        self.dynamics = dynamics_obj
        if controller_file is "":
            assert(controller_obj is not None)
            self.controller = controller_obj
        else:
            assert(controller_obj is None)
            self.controller = TFController(network=controller_file)
        # load controller constraints, states from dynamics, and abstract dynamics at some default epsilon
        # controller constraints loaded by default
        self.states = self.dynamics.states 
        self.next_states = self.dynamics.next_states
        # if self.dynamics.abstract_constraints: # if not empty
        #     self.set_constraints()
        # else: # empty
        #     self.abstract(epsilon)
        self.set_constraints() # TODO: comment out. just for testing.
    
    def abstract(self, initial_set, epsilon, CEx=None):
        """
        Abstract dynamics.
        """
        self.dynamics.abstract(initial_set, epsilon=epsilon)
        self.set_constraints()

    def set_constraints(self):
        """
        Fill self.constraints list with appropriate, up-to-date constraints
        """
        self.constraints = self.dynamics.abstract_constraints + self.controller.constraints
        self.match_io()
    
    def match_io(self):
        """
        precondition: controller and dynamics exist and are valid.
        Matches inputs and outputs of controller and dynamics.
        Adds equality constraint(s).
        """ 
        # match dynamics.states to controller inputs?
        c1 = matrix_equality_constraint(self.dynamics.states, self.controller.state_inputs)
        # match controller output to dynamics control input. 
        c2 = matrix_equality_constraint(self.controller.control_outputs, self.dynamics.control_inputs)
        self.constraints.extend([c1,c2])


class TransitionSystem:
    def __init__(self, states=[], initial_set={}, transition_relation=TransitionRelation()):
        self.states = states 
        self.initial_set = initial_set # set of inequalities over states
        # e.g. init_set = {"x": (0, 5), "theta": (-np.pi/4, np.pi/4)}
        self.transition_relation = transition_relation # object of class TransitionRelation

    # def simple_next(self, state):
    #     # return names of next state variables x'
    #     # return term and mapping?
    #     # return variable x' and then 
    #     return substitute(self.states[state], state)

class MyTransitionSystem(TransitionSystem):
    """
    For now, the interface is as follows:
    states              - list of stringnames of states e.g. ["x", "theta"]
    initial_set         - dictionary mapping stringnames of states to 'boxes' e.g. {"x": (0, 5), "theta": (-np.pi/4, np.pi/4)}
    transition_relation - object derived from TransitionRelation class that defines how e.g. x' and theta' are derived
                            from x and theta

    Future goal:
    - support input boxes as well as plyhedron constraints for inputs. e.g. if given a polyhedon for input set, apply that
    as a constraint over inputs but then find smallest bounding box around that polyhedron and use this bounding box for
    overappoximation algo
    """
    def __init__(self, states=[], initial_set={}, transition_relation=TransitionRelation()):
        super().__init__(states=states, initial_set=initial_set, transition_relation=transition_relation)
    
    def abstract(self, epsilon, CEx=None):
        self.transition_relation.abstract(self.initial_set, epsilon, CEx=CEx)

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

# property
class Property():
    """
    Property that you want to hold.
    For now, box set over state variables.
    Future: linear constraints on outputs OR box set.
    """
    def __init__(self, arg):
        pass

    def complement(self):
        # return complement of desired property
        pass

# property
class ConstraintProperty():
    """
    Property that you want to hold. A list of constraints in CNF.
    """
    def __init__(self, c):
        self.constraints = c

    def complement(self):
        # return complement of desired property
        # for now, only support single constraint
        assert(len(self.constraints) == 1)
        c = self.constraints[0]
        if c.type == ConstraintType('GREATER'):
            ccomp = Constraint(ConstraintType('LESS_EQ'))
            ccomp.monomials = c.monomials
        else:
            ccomp = None
            raise NotImplementedError
        self.constraint_complements = [ccomp]
        # for future, a code sketch:
        # complements = []
        # for c in self.constraints:
        #     complement.append(self.negate(c)) 
        # then disjunct all the complements using MaxConstraint

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
            timed_vars = [v+"@"+str(t) if not isprimed(v) else v[:-1]+"@"+str(t+1) for v in vars_in_c]
            return substitute_Mc(c, dict(zip(vars_in_c, timed_vars)))
        elif isinstance(c, Constraint):
            vars_in_c = [m.var for m in c.monomials]
            timed_vars = [v+"@"+str(t) if not isprimed(v) else v[:-1]+"@"+str(t+1) for v in vars_in_c]
            return substitute_c(c, dict(zip(vars_in_c, timed_vars)))
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
        self.solver.assert_constraints(self.unroller.at_time_property(self.prop, t))

        return self.solver.check_sat() # result, values, stats

    def check_invariant_until(self, time):
        """
        Check invariant property p holds until time t
        """
        # For now "incremental" sort of. 
        for i in range(time):
            result, values, stats = self.step_invariant(i) # checks that property holds at i
            if not (result == Result.UNSAT):
                print("Property may not hold at time ", i)
                return result
        return Result.UNSAT ## TODO: make sure this is correct
    
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
    Counter Example Guided Abstraction Refinement.
    algo object with transi
    """    
    # while property has not been verified and the maximum number of iterations
    # has not yet been exceeded, refine and try again
    #       algo.transition_system.abstract(epsilon, CEx)
    # if property does not hold, return CEx
    # else if it does hold, return :HOLDS
    pass


    
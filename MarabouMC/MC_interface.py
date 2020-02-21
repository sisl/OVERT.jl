#from enum import Enum
#import tensorflow as tf
# from maraboupy import *
from MC_constraints import Constraint, ConstraintType
from MC_TF_parser import TFConstraint
from Constraint_utils import matrix_equality_constraint, equality_constraint

class Result(Enum):
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    
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
        super.__init__(self)
        """
        load constraints by parsing network file
        """
        self.tfconstraintobj = TFConstraint(filename=network_file, sess=tf_sess, inpuNames=inputNames, outputName=outputName)
        self.constraints = self.tfconstraintobj.constraints
        self.state_inputs = self.tfconstraintobj.inputVars
        self.control_outputs = self.tfconstraintobj.outputVars
        self.relus = self.tfconstraintobj.relus

class Dynamics:
    def __init__(self, fun, states, controls):
        self.fun = fun # python function representing the dynamics
        self.states = states
        self.control_inputs = controls
        self.next_states = [x+"'" for x in states] # [x,y,z] -> [x', y', z']
        self.constraints = [] # constraints over the states and next states

class OVERTDynamics(Dynamics):
    def __init__(self, input_file=None):
        super(self).__init__()
        self.abstract_constraints = [] # constraints over the states and next states, linearized

    def abstract(self, initial_set, epsilon=1-e1):
        """
        Convert dynamics constraints from nonlinear to linearized using OVERT and epsilon.
        """
        pass
        # fill self.abstract_constraints
        # add constraints matching self.next_states and what comes out of abstraction generator

class ControlledTranstionRelation(TransitionRelation):
    def __init__(self, controller_file=None, dynamics_file=None, epsilon=1-6):
        super(self).__init__() # states, next states, constraints containers
        self.controller_file = controller_file
        self.dynamics_file = dynamics_file
        self.epsilon = epsilon

class TFControlledOVERTTransitionRelation(ControlledTranstionRelation):
    
    def __init__(self, dynamics_obj, controller_file="", controller_obj=None, epsilon=1e-6):
        """
        Constructor takes objs XOR filenames.
        """
        super(self).__init__()
        self.dynamics = dynamics_obj
        if controller_file is "":
            assert(controller_obj is not None)
            self.controller = controller_obj
        else:
            assert(controller_obj is None)
            self.controller = TFController(network=controller_file)
        # load controller constraints, states from dynamics, and abstract dynamics at some default epsilon
        # controller constraints loaded by default
        self.states = dynamics.states 
        self.next_states = dynamics.next_states
        # if self.dynamics.abstract_constraints: # if not empty
        #     self.set_constraints()
        # else: # empty
        #     self.abstract(epsilon)
    
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
        c2 = matrix_equality_constraint(self.controller.control_output, self.dynamics.control_inputs)
        self.constraints.extend([c1,c2])


class TransitionSystem:
    def __init__(self, states=[], initial_set={}, transition_relation=TransitionRelation()):
        self.states = states 
        self.initial_set = initial_set # set of inequalities over states
        self.transition_relation = transition_relation # object of class TransitionRelation

    # def simple_next(self, state):
    #     # return names of next state variables x'
    #     # return term and mapping?
    #     # return variable x' and then 
    #     return substitute(self.states[state], state)

class MyTransitionSystem(TransitionSystem):
    def __init__(self, states=[], initial_set={}, transition_relation=TransitionRelation()):
        super.__init__(self)
    
    def abstract(self, epsilon, CEx=None):
        self.transition_relation.abstract(self.initial_set, epsilon, CEx=CEx)

def substitute(c: Constraint, mapping): 
    """
    substitute(x+y<0, [x => x@1, y => y@1])
    Used in unroller.
    """
    new_constraint = Constraint(c.type)
    new_constraint.scalar = c.scalar
    new_constraint.monomials = c.monomials
    for mono in new_constraint.monomials:
        mono[1] = mapping[mono[1]]
    return new_constraint
    
# solver
class MarabouWrapper():
    """
    A class which converts to the maraboupy interface.
    """
    def __init__(self):
        # initialize "clean" query?
        pass

    def clear(self):
        # init new marabou query
        pass

    def assert(self, constraints):
        # store constraints in some sort of internal representation
        pass

    def assert_init(self, set, states):
        # assert states /in InitSet set
        pass

    def convert(self, constraints):
        """
        Takes in a set of constraints and converts them to a format that Maraboupy understands.
        """
        pass

    def check_sat():
        # call convert to convert internal representation of timed contraints to marabou vars + ineqs
        pass

# property
class Property():
    """
    Property that you want to hold.
    """
    def __init__(self):
        pass

    def complement(self):
        # return complement of desired property
        pass

class Unroller():
    def __init__(self):
        pass
        self.cache = None # dictionary from original to timed version (may be helpful)

    def at_time_constraint(self, c: Constraint, tstep):
        """
        at_time(x+y'<0, 2)
            return x@2 + y@3 < 0
        The timestep passed corresponds to the _unprimed_ variable.
        """
        pass

    def at_time_relation(self, TR: TransitionRelation, tstep):
        # for every constraint in the transition relation, call at_time and do subsitution
        pass

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

# BMC , a model checking algo
class BMC():
    def __init__(self, ts, prop_file: str, solver=None):
        self.transition_sys = ts 
        self.prop = prop_file 
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
        solver.clear() # new query
        # assert that state begins in init set
        solver.assert_init(self.transition_sys.initial, self.transition_sys.states)

        # unroll transition relation for time 0 to t
        for j in range(t):
            solver.assert(self.unroller.at_time_relation(self.transition_sys.transition_relation, j))
        
        # assert complement(property) at time t
        solver.assert(self.unroller.at_time_relation(self.prop.complement(), t))

        return solver.check_sat()

    def check_invariant_until(self, time):
        """
        Check invariant property p holds until time t
        """
        # For now "incremental" sort of. 
        for i in range(time):
            if not self.step_invariant(i): # checks that property holds at i
                print("Property does not hold at time ", i)
                return Result.SAT 
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


    
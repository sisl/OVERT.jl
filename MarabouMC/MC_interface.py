from enum import Enum
import tensorflow as tf
# from maraboupy import *

class ConstraintType(Enum):
    EQUALITY = 0
    LESS_EQ = 1
    LESS = 2
    GREATER_EQ = 3
    GREATER = 4

class Result(Enum):
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2

class Constraint:
    def __init__(self, eqtype: ConstraintType):
        """
        sum_i(monomial_i) ConstraintType scalar
        e.g. 5x + 3y <= 0
        """
        self.type = eqtype
        self.monomials = [] # list of tuples of (coeff, var)
        self.scalar = 0
    
class TransitionRelation:
    def __init__(self):
        states = []
        next_states = []
        constraints = []

class Controller:
    def __init__(self, network=None):
        self.network = network
        self.equations = []

class TFController(Controller):
    def __init__(self, network=None):
        super.__init__(self)
        self.load()

    def load(self):
        """
        load equations by parsing network file
        """
        pass

class Dynamics:
    def __init__(self, input_file=None):
        self.input_file = input_file
        self.equations = [] # equations over the states and next states
        self.load()

    def load(self):
        """
        load equations by parsing dynamics file
        """
        pass

class ControlledTranstionRelation(TransitionRelation):
    def __init__(self, controller_file=None, dynamics_file=None, epsilon=1-6):
        super(self).__init__()
        self.controller_file = controller_file
        self.dynamics_file = dynamics_file
        self.epsilon = epsilon

class TFControlledTransitionRelation(ControlledTranstionRelation):
    def __init__(self, controller_file="", dynamics_file="", epsilon=1-6):
        super(self).__init__()
        self.controller = TFController(network=controller_file)
        self.dynamics = Dynamics(input_file=dynamics_file)
    
    def __init__(self, controller_obj, dynamics_obj, epsilon=1e-6):
        """
        Alternate constructor that takes objs instead of filenames.
        """
        self.controller_file = ""
        self.dynamics_file = ""
        self.epsilon = epsilon
        self.controller = controller_obj
        self.dynamics = dynamics_obj

    def convert_dynamics(self):
        """
        Convert dynamics constraints from nonlinear to linearized using OVERT and epsilon
        TODO: maybe move elsewhere? to unroller maybe?
        """
        pass

class TransitionSystem:
    def __init__(self, states=Dict(), initial_set=set(), transition_relation=TransitionRelation()):
        self.transition_relation = transition_relation # object of class TransitionRelation
        self.states = states  # dictionary mapping current states to next states
        self.initial = initial_set # set of inequalities over states

    # def simple_next(self, state):
    #     # return names of next state variables x'
    #     # return term and mapping?
    #     # return variable x' and then 
    #     return substitute(self.states[state], state)


def substitute(c: Constraint, mapping): 
    """
    substitute(x+y<0, [x => x@1, y => y@1])
    Used in unroller.
    """
    pass
    
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
        Check that property p holds at time t. Assumes it holds at 1:t-1
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
            


    
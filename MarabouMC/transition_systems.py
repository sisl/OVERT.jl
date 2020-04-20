# transition system component classes
from MC_TF_parser import TFConstraint
from Constraint_utils import matrix_equality_constraint
from MC_constraints import ReluConstraint, Constraint, ConstraintType, Monomial
from MC_Keras_parser import KerasConstraint
import numpy as np

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

class KerasController(Controller):
    def __init__(self, keras_model=None):
        super().__init__()
        """
        load constraints by parsing network file
        """
        self.kerasconstraintobj = KerasConstraint(model=keras_model)
        self.constraints = self.kerasconstraintobj.constraints
        self.state_inputs = np.array(self.kerasconstraintobj.model_input_vars).reshape(-1, 1)
        self.control_outputs = np.array(self.kerasconstraintobj.model_output_vars).reshape(-1, 1)
        self.relus = [c for c in self.constraints if isinstance(c, ReluConstraint)]

class Dynamics:
    def __init__(self, fun, states, controls):
        self.fun = fun # python function representing the dynamics
        self.states = np.array(states).reshape(-1,1)
        self.control_inputs = np.array(controls).reshape(-1,1)
        self.next_states = np.array([x+"'" for x in states.flatten()]).reshape(self.states.shape) # [x,y,z] -> [x', y', z']
        self.constraints = [] # constraints over the states and next states

class OVERTDynamics(Dynamics):
    def __init__(self, fun=None, overt_objs=None, dt=0.1):
        self.overt_objs = overt_objs
        self.assert_overt_objs()
        states_vars = np.array(self.overt_objs[0].state_vars).reshape(-1, 1)
        controls_vars = np.array(self.overt_objs[0].control_vars).reshape(-1, 1)
        super().__init__(fun, states_vars, controls_vars)

        self.overt_constraints = [] # constriants introduced by overt
        self.euler_constraints = [] # constraints introduced by euler integration.
        self.get_overt_constraints()
        self.get_euler_constraint(dt)
        self.constraints = self.overt_constraints + self.euler_constraints

    def assert_overt_objs(self):
        for i in range(len(self.overt_objs)-1):
            assert set(self.overt_objs[i].state_vars) == set(self.overt_objs[i + 1].state_vars), "overt_objs should have same state variables"
            assert set(self.overt_objs[i].control_vars) == set(self.overt_objs[i + 1].control_vars), "overt_objs should have same control variables"
            assert len(self.overt_objs) == len(self.overt_objs[i].state_vars) // 2, "with %d state variables, there should be %d overt objects" %(len(self.state_vars), len(self.overt_objs))

    def get_overt_constraints(self):
        for overt_obj in self.overt_objs:
            self.overt_constraints += overt_obj.constraints

    def get_euler_constraint(self, dt):
        """
        Add additional constraint for euler integration.
        It is assumed the format of state vector is like [x1, x2, dx1, dx2]
        """
        ndim = len(self.states) // 2
        for i in range(ndim):
            # x_next = x + dt*u
            c1 = Constraint(ConstraintType('EQUALITY'))
            c1.monomials = [Monomial(1, self.states[i, 0]), Monomial(dt, self.states[i + ndim, 0]), Monomial(-1, self.next_states[i, 0])]
            self.euler_constraints.append(c1)

            # u_next = u + dt*a
            c2 = Constraint(ConstraintType('EQUALITY'))
            c2.monomials = [Monomial(1, self.states[i + ndim, 0]), Monomial(dt, self.overt_objs[i].output_vars[0]), Monomial(-1, self.next_states[i + ndim, 0])]
            self.euler_constraints.append(c2)
        # fill self.abstract_constraints
        # add constraints matching self.next_states and what comes out of abstraction generator

class ControlledTranstionRelation(TransitionRelation):
    """
    A transition relation with a controller and dynamics.
    """
    def __init__(self, controller_file="", controller_obj=None, dynamics_obj=None, epsilon=1-6):
        super().__init__() # states, next states, constraints containers
        self.controller_file = controller_file
        self.controller = controller_obj
        self.dynamics = dynamics_obj
        self.states = self.dynamics.states 
        self.next_states = self.dynamics.next_states
        self.epsilon = epsilon
        self.set_constraints()
    
    def set_constraints(self):
        """
        Fill self.constraints list with appropriate, up-to-date constraints.
        Overload in derived classes if necessary.
        """
        self.constraints = self.dynamics.constraints + self.controller.constraints
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

class TFControlledTransitionRelation(ControlledTranstionRelation):
    """
    A transition relation with a tf controller and dynamics.
    """
    def __init__(self, dynamics_obj=None, controller_file="", controller_obj=None,  epsilon=1-6):
        if controller_file is "":
            assert(controller_obj is not None)
            controller = controller_obj
        else:
            assert(controller_obj is None)
            controller = TFController(network_file=controller_file)
        super().__init__(controller_file=controller_file, controller_obj=controller, dynamics_obj=dynamics_obj, epsilon=epsilon)


class TFControlledOVERTTransitionRelation(TFControlledTransitionRelation):
    
    def __init__(self, dynamics_obj, controller_file="", controller_obj=None, epsilon=1e-6):
        """
        Constructor takes objs XOR filenames.
        """
        # inherited constructor should call self.set_constraints()
        super().__init__(controller_file=controller_file, controller_obj=controller_obj, dynamics_obj=dynamics_obj, epsilon=epsilon)
    
    def abstract(self, initial_set, epsilon, CEx=None):
        """
        Abstract dynamics.
        """
        self.dynamics.abstract(initial_set, epsilon=epsilon) # populates self.dynamics.abstract_constraints
        self.set_constraints()

    def set_constraints(self):
        """
        Fill self.constraints list with appropriate, up-to-date constraints.
        WARNING: abstract_constraints will be EMPTY until you call self.abstract()
        """
        print("calling constructor for TFControlledOVERTTransitionRelation")
        self.constraints = self.dynamics.abstract_constraints + self.controller.constraints
        self.match_io()


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

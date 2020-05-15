# transition system component classes
from MC_TF_parser import TFConstraint
from Constraint_utils import matrix_equality_constraint
from MC_constraints import ReluConstraint, Constraint, ConstraintType, Monomial
from MC_Keras_parser import KerasConstraint
from overt_to_python import OvertConstraint
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
    def __init__(self, keras_model=None, cap_values=None):
        super().__init__()
        """
        load constraints by parsing network file
        """
        self.kerasconstraintobj = KerasConstraint(model=keras_model)
        if cap_values is not None:
            self.kerasconstraintobj.cap(cap_values[0], cap_values[1])
        self.constraints = self.kerasconstraintobj.constraints
        self.state_inputs = np.array(self.kerasconstraintobj.model_input_vars).reshape(-1, 1)
        self.control_outputs = np.array(self.kerasconstraintobj.model_output_vars).reshape(-1, 1)
        self.relus = [c for c in self.constraints if isinstance(c, ReluConstraint)]

class Dynamics:
    def __init__(self, states, controls):
        self.states = np.array(states).reshape(-1,1)
        self.control_inputs = np.array(controls).reshape(-1,1)
        self.next_states = np.array([x+"'" for x in states.flatten()]).reshape(self.states.shape) # [x,y,z] -> [x', y', z']
        self.constraints = [] # constraints over the states and next states

# class OVERTDynamics(Dynamics):
#     """
#     This object takes two inputs:
#     - overt_objs: a list of OvertConstraint objects.
#     - time_update_dict: a dictionary with following key-values:
#         - type: "continuous" : x_{t+1} = x_t + dt*dx_t
#                 "discrete" : x_{t+1} is explicitly given
#         - dt: dt for continuous
#         - map: a dictionary with key-values= x_t => dx_t for continuous and x_{t+1} => x_{t+1} for discrete.
#     """
#     def __init__(self, overt_objs, time_update_dict):
#         # check all overt_objs are OvertConstant objects and have same state and control variables.
#         self.assert_overt_objs(overt_objs)
#         super().__init__(np.array(overt_objs[0].state_vars).reshape(-1, 1),
#                          np.array(overt_objs[0].control_vars).reshape(-1, 1))
#
#         # merge overt objects constrants
#         self.overt_constraints = []
#         for obj in overt_objs:
#             self.overt_constraints += obj.constraints
#
#         # add additional constraints for time advancement
#         self.euler_constraints = []
#         self.setup_euler_constraint(time_update_dict) # constraints introduced by time update
#         self.constraints = self.overt_constraints + self.euler_constraints
#
#     @staticmethod
#     def assert_overt_objs(overt_objs):
#         assert type(overt_objs) == type([]), "overt_objs has to be a list."
#         for i in range(len(overt_objs)-1):
#             assert set(overt_objs[i].state_vars) == set(overt_objs[i + 1].state_vars), "overt_objs should have same state variables"
#             assert set(overt_objs[i].control_vars) == set(overt_objs[i + 1].control_vars), "overt_objs should have same control variables"
#
#     def setup_euler_constraint(self, time_update_dict):
#         assert set(time_update_dict["map"].keys()) == set(self.states.reshape(-1)),  "time_update_dict[map] keys should be state variables."
#         if time_update_dict["type"] == "continuous":
#             dt = time_update_dict["dt"]
#             for x, next_x in zip(self.states.reshape(-1), self.next_states.reshape(-1)):
#                 dx = time_update_dict["map"][x]
#                 c = Constraint(ConstraintType('EQUALITY'))
#                 c.monomials = [Monomial(1, x), Monomial(dt, dx), Monomial(-1, next_x)]
#                 self.euler_constraints.append(c)
#         elif time_update_dict["type"] == "discrete":
#             raise(NotImplementedError())
#
#         # fill self.abstract_constraints
#         # add constraints matching self.next_states and what comes out of abstraction generator

class OvertDynamics(Dynamics):
    """
    This object three inputs:
    - overt_objs: an instance of OvertConstraint object.
    - dx_vec: vector of dx
    - dt: dt for continuous              # this is confusing 
    """
    def __init__(self, overt_objs, dx_vec, dt):
        super().__init__(np.array(overt_objs.state_vars).reshape(-1, 1),
                         np.array(overt_objs.control_vars).reshape(-1, 1))

        # overt objects constrants
        self.overt_constraints = overt_objs.constraints

        # add additional constraints for time advancement
        self.euler_constraints = []
        self.setup_euler_constraint(dx_vec, dt) # constraints introduced by time update
        self.constraints = self.overt_constraints + self.euler_constraints

    def setup_euler_constraint(self, dx_vec, dt):
        for x, dx, next_x in zip(self.states.reshape(-1), dx_vec, self.next_states.reshape(-1)):
                c = Constraint(ConstraintType('EQUALITY'))
                c.monomials = [Monomial(1, x), Monomial(dt, dx), Monomial(-1, next_x)]
                self.euler_constraints.append(c)
    
    def setup_continuous_constraints(self):
        # for handling continuous time
        pass

class NonlinearDynamics(Dynamics):
    """
    takes argument indicating true dynamics function e.g. SinglePendulum()
    dx_vec
    dt (or 0 for continuous)
    """
    def __init__(self, nonlinear_dynamics, dt):
        super().__init__(np.array(nonlinear_dynamics.states).reshape(-1,1),
                        np.array(nonlinear_dynamics.control_inputs).reshape(-1,1)
                        )
        self.dt = dt
        self.set_dynamics(nonlinear_dynamics)
        if dt > 0:
            self.setup_euler_constraints()
        else:
            raise NotImplementedError     # continuous time not imlemented
    
    def set_dynamics(self, nonlinear_dynamics):
        self.dynamics = nonlinear_dynamics # store for safekeeping
        self.dx = nonlinear_dynamics.dx
        self.constraints += nonlinear_dynamics.dx_constraints
    
    def setup_euler_constraints(self):
        for x, dx, next_x in zip(self.states.reshape(-1), self.dx, self.next_states.reshape(-1)):
                # next_x = x + dx*dt
                c = Constraint(ConstraintType('EQUALITY'))
                c.monomials = [Monomial(1, x), Monomial(self.dt, dx), Monomial(-1, next_x)]
                self.constraints.append(c)

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


def constraint_variable_to_interval(variable, LB, UB):
    p1 = Constraint(ConstraintType('GREATER'))
    p1.monomials = [Monomial(1, variable)]
    p1.scalar = LB # 0 #
    #
    p2 = Constraint(ConstraintType('LESS'))
    p2.monomials = [Monomial(1, variable)]
    p2.scalar = UB
    return [p1, p2]
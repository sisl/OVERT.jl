
import os
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "/home/amaleki/Downloads/Marabou/")
print(sys.path)

import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem, OVERTDynamics
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from IterativeReachability import ReachabilityIterator
#from MC_simulate import simulate_double_pend

# This is trained controller with lqr data.
model = load_model("../OverApprox/models/double_pend_nn_controller_lqr_data.h5")
# This a untrained controller
# model = load_model("../OverApprox/models/double_pend_controller_nn_not_trained.h5")
# super simple controller
# model = load_model("/home/amaleki/Downloads/test_6_linear.h5")
controller = KerasController(keras_model=model)

# create overt dynamics objects
overt_obj = OvertConstraint("../OverApprox/models/double_pendulum_overt.h5")

# setup states, control and dynamics variables.
states = overt_obj.state_vars
theta1 = states[0]
theta2 = states[1]
theta1d = states[2]
theta2d = states[3]


controls = overt_obj.control_vars
acceleration_1 = overt_obj.output_vars[0]
acceleration_2 = overt_obj.output_vars[1]

double_pendulum_dynamics = Dynamics(np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
next_states = double_pendulum_dynamics.next_states.reshape(4,)


dt = 0.01

time_update_dict = {"dt": dt,
                    "type": "continuous",
                    "map": {states[0]: states[2], states[1]: states[3], states[2]: acceleration_1, states[3]: acceleration_2}
                    }
double_pendulum_dynamics = OVERTDynamics([overt_obj], time_update_dict)


###################################
init_set = {theta1: (0.5, 0.6), theta2: (0.5, 0.6), theta1d: (-0.5, 0.5), theta2d: (-0.5, 0.5)}
ri = ReachabilityIterator(model, double_pendulum_dynamics, init_set, alpha=1.5, cap_values=[[-2., -2.],[2., 2.]])
ri.run(6)
for h in ri.init_history:
    print(h)

####################################
print(dsdsfd)


# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=double_pendulum_dynamics,
                                        controller_obj=controller)

# initial set
init_set = {theta1: (0.5, 0.6), theta2: (0.5, 0.6), theta1d: (-0.5, 0.5), theta2d: (-0.5, 0.5)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

print(len([c for c in double_pendulum_dynamics.constraints if isinstance(c, ReluConstraint)]))
print(len([c for c in controller.constraints if isinstance(c, ReluConstraint)]))

# solver
solver = MarabouWrapper()

def constraint_variable_to_interval(variable, LB, UB):
    p1 = Constraint(ConstraintType('GREATER'))
    p1.monomials = [Monomial(1, variable)]
    p1.scalar = LB # 0 #
    #
    p2 = Constraint(ConstraintType('LESS'))
    p2.monomials = [Monomial(1, variable)]
    p2.scalar = UB
    return [p1, p2]


# property th1 < some_number
prop_list = []
prop_list += constraint_variable_to_interval(theta1, .2, .8)
prop_list+=constraint_variable_to_interval(theta2, .2, .8)
prop_list+=constraint_variable_to_interval(theta1d, -.6, .6)
prop_list+=constraint_variable_to_interval(theta2d, -.6, .6)

prop = ConstraintProperty(prop_list)

# algo
ncheck_invariant = 2
algo = BMC(ts=ts, prop=prop, solver=solver)
result = algo.check_invariant_until(ncheck_invariant)

# random runs to give intuition to MC result
# n_simulation = 10000
# print("Now running %d simulations: " %n_simulation, end="")
# simulate_double_pend(prop, n_simulation, ncheck_invariant, model, dt, init_set, states)

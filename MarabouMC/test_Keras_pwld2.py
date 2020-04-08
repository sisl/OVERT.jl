# test TF network controller and pwl plant


import os
import sys
#sys.path.insert(0, "/home/amaleki/Dropbox/stanford/Python/Marabou/")
#print(sys.path)

import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC

# create controller object with a keras model
model = load_model("../OverApprox/models/double_pend_controller_nn_not_trained.h5")
controller = KerasController(keras_model=model)

# create overt dynamics objects
overt_obj_1 = OvertConstraint("../OverApprox/models/double_pend_acceleration_1_overt.h5")
overt_obj_2 = OvertConstraint("../OverApprox/models/double_pend_acceleration_2_overt.h5")
#overt_obj = OvertConstraint("../OverApprox/src/up.h5")
states = overt_obj_1.state_vars
controls = overt_obj_1.control_vars
assert(all(states == overt_obj_2.state_vars))
assert(all(controls == overt_obj_2.control_vars))
acceleration_1 = overt_obj_1.output_vars[0]
acceleration_2 = overt_obj_2.output_vars[0]

double_pendulum_dynamics = Dynamics(None, np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
next_states = double_pendulum_dynamics.next_states.reshape(4,)

print(states, controls, acceleration_1, acceleration_2, next_states)

# timestep = 0.1
dt = 0.1

# x1_next = x1 + dt*u1
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, states[0]), Monomial(dt, states[2]), Monomial(-1, next_states[0])]
print(c1.monomials)


# x2_next = x2 + dt*u2
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1, states[1]), Monomial(dt, states[3]), Monomial(-1, next_states[1])]
print(c2.monomials)

# u1_next = u1 + dt*a1
c3 = Constraint(ConstraintType('EQUALITY'))
c3.monomials = [Monomial(1, states[2]), Monomial(dt, acceleration_1), Monomial(-1, next_states[2])]
print(c3.monomials)

# u2_next = u2 + dt*a2
c4 = Constraint(ConstraintType('EQUALITY'))
c4.monomials = [Monomial(1, states[3]), Monomial(dt, acceleration_2), Monomial(-1, next_states[3])]
print(c4.monomials)

dynamics_constraints = [c1, c2, c3, c4]
dynamics_constraints += overt_obj_1.eq_list + overt_obj_1.ineq_list + overt_obj_1.relu_list + overt_obj_1.max_list
dynamics_constraints += overt_obj_2.eq_list + overt_obj_2.ineq_list + overt_obj_2.relu_list + overt_obj_2.max_list
double_pendulum_dynamics.constraints = dynamics_constraints

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=double_pendulum_dynamics,
                                        controller_obj=controller)

# initial set
init_set = {states[0]: (0., 0.1), states[1]: (0., 0.1), states[2]: (-1., 1.), states[3]: (-1., 1.)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

# property x< 0.105, x' < 0.2
p = Constraint(ConstraintType('LESS'))
p.monomials = [Monomial(1, states[0])]
p.scalar = 0.101 # 0 #
prop = ConstraintProperty([p])

# algo
algo = BMC(ts = ts, prop = prop, solver=solver)
algo.check_invariant_until(10)

# random runs to give intuition to MC result
# for i in range(10):
#     x = np.random.rand()*(2 - 1.1) + 1.1
#     print("x@0=", x)
#     y = np.random.rand()*(1 - -1) + -1
#     for j in range(3):
#         state = np.array([x,y]).flatten().reshape(-1,1)
#         u = np.maximum(0, W2@(np.maximum(0,W1@state + b1)) + b2)
#         #x' = relu(x + u)
#         x = max(0, x + u.flatten()[0])
#         print("x@",j+1,"=", x)
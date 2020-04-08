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
model = load_model("../OverApprox/models/single_pend_nn_ilqr_data.h5")
controller = KerasController(keras_model=model)

# create overt dynamics objects
overt_obj = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")
#overt_obj = OvertConstraint("../OverApprox/src/up.h5")
states = overt_obj.state_vars
controls = overt_obj.control_vars
acceleration = overt_obj.output_vars[0]

single_pendulum_dynamics = Dynamics(None, np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
next_states = single_pendulum_dynamics.next_states.reshape(2,)

print(states, controls, acceleration, next_states)

# x_next = x + dt*u
dt = 0.1
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, states[0]), Monomial(dt, states[1]), Monomial(-1, next_states[0])]
print(c1.monomials)

# u_next = u + dt*a
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1, states[1]), Monomial(dt, acceleration), Monomial(-1, next_states[1])]
print(c2.monomials)

single_pendulum_dynamics.constraints = [c1, c2] # + overt_obj.constraint

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=single_pendulum_dynamics,
                                        controller_obj=controller)

# initial set
init_set = {states[0]: (0., 0.1), states[1]: (-1., 1.)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

# property x< 0.105, x' < 0.2
p = Constraint(ConstraintType('LESS'))
p.monomials = [Monomial(1, states[0])]
p.scalar = 0.101 # 0 #
prop = ConstraintProperty([p])


p.monomials = [Monomial(1, states[0])]
p.scalar = 0.101 # 0 #
prop = ConstraintProperty([p])

# algo
algo = BMC(ts = ts, prop = prop, solver=solver)
algo.check_invariant_until(20)

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
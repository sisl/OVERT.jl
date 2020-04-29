# test keras network controller and pwl plant

import sys
sys.path.append('..')
import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem, OVERTDynamics
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial, MaxConstraint, ReluConstraint
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from MC_simulate import simulate
from funs import single_pendulum


# create controller object with a keras model
# good controller
model = load_model("../OverApprox/models/single_pend_nn_controller_ilqr_data.h5")
#model = load_model("../OverApprox/models/single_pend_nn_controller_lqr_data.h5")
# bad controller
# model = load_model("../OverApprox/models/single_pend_controller_nn_not_trained.h5")
# bad yet super simple model
# model = load_model("/home/amaleki/Downloads/test_2_linear.h5")
# model = load_model("/home/amaleki/Downloads/test_3_linear.h5")
# model = load_model("/home/amaleki/Downloads/test_2_relu.h5")
# model = load_model("/home/amaleki/Downloads/test_3_relu.h5")

controller = KerasController(keras_model=model)
print(controller.constraints)
# rewrite to make a simple controller that is always equal to 1.0
# monomial_list = [Monomial(1, controller.control_outputs[0][0])]
# fake_constraint = [Constraint(ConstraintType('EQUALITY'), monomial_list, 1.0)]
# controller.constraints = fake_constraint
# controller.relus = []

# create overt dynamics objects
overt_obj = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")
for c in overt_obj.constraints:
    print(c)

#overt_obj = OvertConstraint("../OverApprox/src/up.h5")
states = overt_obj.state_vars
controls = overt_obj.control_vars
acceleration = overt_obj.output_vars[0]

print(states, " ", controls, " ", acceleration)

dt = 0.1
time_update_dict = {"dt": dt,
                    "type": "continuous",
                    "map": {states[0]: states[1], states[1]: acceleration}
                    }

single_pendulum_dynamics = OVERTDynamics([overt_obj], time_update_dict)

# single_pendulum_dynamics = Dynamics(None, np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
# single_pendulum_dynamics.constraints = overt_obj.constraints.copy()
# next_states = single_pendulum_dynamics.next_states.reshape(2,)
#
# # x_next = x + dt*u
# dt = 0.1
# c1 = Constraint(ConstraintType('EQUALITY'))
# c1.monomials = [Monomial(1, states[0]), Monomial(dt, states[1]), Monomial(-1, next_states[0])]
#
# # u_next = u + dt*a
# c2 = Constraint(ConstraintType('EQUALITY'))
# c2.monomials = [Monomial(1, states[1]), Monomial(dt, acceleration), Monomial(-1, next_states[1])]
#
# single_pendulum_dynamics.constraints += [c1, c2]

# print("single pendulum dynamics constraints = ", len(single_pendulum_dynamics.constraints))
# print("controler constraints = ", len(controller.constraints))

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=single_pendulum_dynamics,
                                        controller_obj=controller)

# initial set
x1_init_set = (0.5, 1)
x2_init_set = (-0.5, 0.5)
init_set = {states[0]: x1_init_set, states[1]: x2_init_set}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

prop_list =[]
p1 = Constraint(ConstraintType('GREATER'))
p1.monomials = [Monomial(1, states[0])]
p1.scalar = 0.3
prop_list.append(p1)

p2 = Constraint(ConstraintType('LESS'))
p2.monomials = [Monomial(1, states[0])]
p2.scalar = 1.15
prop_list.append(p2)

# p3 = Constraint(ConstraintType('GREATER'))
# p3.monomials = [Monomial(1, states[1])]
# p3.scalar = -1.1
# prop_list.append(p3)
# #
# p4 = Constraint(ConstraintType('LESS'))
# p4.monomials = [Monomial(1, states[1])]
# p4.scalar = 1.49
# prop_list.append(p4)

prop = ConstraintProperty(prop_list)


# algo
ncheck_invariant = 4
algo = BMC(ts=ts, prop=prop, solver=solver)
result = algo.check_invariant_until(ncheck_invariant)

# random runs to give intuition to MC result
n_simulation = 10000
simulate(single_pendulum, prop, model, init_set, states, n_simulation, ncheck_invariant, dt)
#simulate_single_pend(prop, n_simulation, ncheck_invariant, model, dt, init_set, states)

# test TF network controller and pwl plant


import os
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "/home/amaleki/Downloads/Marabou/")
print(sys.path)

import numpy as np
from keras.models import load_model
from MC_Keras_parser import getNewVariable
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem, OVERTDynamics, constraint_variable_to_interval
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial, MaxConstraint
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from MC_simulate import simulate
from gym_new.car import CarEnv
from IterativeReachability import ReachabilityIterator

controller_file_name = "/home/amaleki/Downloads/car_relu.h5"
model = load_model(controller_file_name)
controller = KerasController(keras_model=model)
# # This is a controller
# controller_file_name = "/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_10/neural_network_controller_1_keras.h5"
# model = load_model(controller_file_name)
# controller = KerasController(keras_model=model)
#
# # HSCC controller has an offset and scale_factor.
# # u = (u-offset)*scale_factor. Here I add these .
# scale_factor, offset = 1.0, 20.0
# u1 = controller.control_outputs[0, 0]
# u2 = controller.control_outputs[1, 0]
# u1_new = getNewVariable("xs")
# u2_new = getNewVariable("xs")
# c1 = Constraint("EQUALITY", monomials=[Monomial(scale_factor, u1), Monomial(-1.0, u1_new)], scalar = scale_factor*offset)
# c2 = Constraint("EQUALITY", monomials=[Monomial(scale_factor, u2), Monomial(-1.0, u2_new)], scalar = scale_factor*offset)
# controller.constraints += [c1, c2]
# controller.control_outputs = np.array([u1_new, u2_new]).reshape(2,1)
#
# # make second controller input bound between -1.5 and 1.5 otherwise tan(u2) blows up.
# # xs1 = -1.5, xs2 = max(u2, xs1), xs3 = -xs2, xs4 = max(xs1, xs3), u2_capped = -xs4
#
# u2 = controller.control_outputs[1, 0]
# xs1 = getNewVariable("xs")
# xs2 = getNewVariable("xs")
# xs3 = getNewVariable("xs")
# xs4 = getNewVariable("xs")
# u2_capped = getNewVariable("xs")
# constraint1 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs1)], scalar = -1.5) # set xs1 = -1.5
# constraint2 = MaxConstraint(varsin=[u2, xs1], varout = xs2) # xs2 = max(u2, xs1)
# constraint3 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs2), Monomial(1.0, xs3)], scalar = 0) # set xs3 = -xs2
# constraint4 = MaxConstraint(varsin=[xs1, xs3], varout = xs4) # xs4 = max(xs1, xs3)
# constraint5 = Constraint("EQUALITY", monomials=[Monomial(1.0, xs4), Monomial(1.0, u2_capped)], scalar = 0) # set xs3 = -xs2
#
# controller.constraints += [constraint1, constraint2, constraint3, constraint4, constraint5]
# controller.control_outputs[1, 0] = u2_capped

# create overt dynamics objects
overt_obj = OvertConstraint("../OverApprox/models/car_dxdt.h5")

# setup states, control and dynamics variables.
states = overt_obj.state_vars
x1 = states[0]
x2 = states[1]
x3 = states[2]
x4 = states[3]

controls = overt_obj.control_vars
dx1 = overt_obj.output_vars[0]
dx2 = overt_obj.output_vars[1]
dx3 = overt_obj.output_vars[2]
dx4 = controls[0]

dt = 0.2
time_update_dict = {"dt": dt,
                    "type": "continuous",
                    "map": {x1: dx1, x2: dx2, x3: dx3, x4: dx4}
                    }
car_dynamics = OVERTDynamics([overt_obj], time_update_dict)


# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=car_dynamics,
                                        controller_obj=controller)

print("# dynamics relu=", len([c for c in car_dynamics.constraints if isinstance(c, ReluConstraint)]))
print("# dynamics max=", len([c for c in car_dynamics.constraints if isinstance(c, MaxConstraint)]))
print("# controllers relu=", len([c for c in controller.constraints if isinstance(c, ReluConstraint)]))



###################################
init_set = {x1: (9.5, 9.6), x2: (-4.5, -4.4), x3: (2.1, 2.12), x4: (1.5, 1.52)}
ri = ReachabilityIterator(model, car_dynamics, init_set, alpha=1.1)
ri.run(6)
for h in ri.init_history:
    print(h)

##########################################3
print(dsdsfd)
solver = MarabouWrapper()

# initial set
init_set = {x1: (9.5, 9.6), x2: (-4.5, -4.4), x3: (2.1, 2.12), x4: (1.5, 1.52)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)


prop_list = []
prop_list += constraint_variable_to_interval(x1, 5, 10)
prop_list += constraint_variable_to_interval(x2, -5, 0)


# [1*xd518 > 9.047619047619047, 1*xd518 < 10.08, 1*xd519 > -4.7250000000000005, 1*xd519 < -4.190476190476191, 1*xd212 > 2.0, 1*xd212 < 2.2260000000000004, 1*xd334 > 1.4285714285714286, 1*xd334 < 1.6758000000000002]
# prop_list += constraint_variable_to_interval(x1, 9.047619047619047, 10.08)
# prop_list += constraint_variable_to_interval(x2, -4.7250000000000005, -4.190476190476191)
# prop_list += constraint_variable_to_interval(x3, 2.0, 2.2260000000000004)
# prop_list += constraint_variable_to_interval(x4, 1.4285714285714286, 1.6758000000000002)

prop = ConstraintProperty(prop_list)

# algo
ncheck_invariant = 3
algo = BMC(ts=ts, prop=prop, solver=solver)

result, value, stats = algo.check_invariant_until(ncheck_invariant)

value_dict = dict(list(value))
for d in range(ncheck_invariant):
    for i, x in enumerate(states):
        v = "%s@%d" %(x, d)
        if v in value_dict:
            print("x_%d at t=%d = %0.3f" %(i, d, value_dict[v]))

value_vals = [v[1] for v in list()]

# random runs to give intuition to MC result
n_simulation = 10000
print("Now running %d simulations: " %n_simulation, end="")
simulate(CarEnv, prop, model, init_set, states, n_simulation, ncheck_invariant, dt, use_env=True)
print(states)
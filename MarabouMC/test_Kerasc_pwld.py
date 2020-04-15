# test keras network controller and pwl plant

import sys
sys.path.append('..')
from gym_new.pendulum_new import Pendulum1Env
import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial, MaxConstraint, ReluConstraint
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from funs import single_pendulum

def property_violated(env, prop):
    for hist in env.history:
        x = hist["x"][0]
        for c in prop.constraints:
            type = c.type.type2str[c.type._type]
            if type == ">":
                if x * c.monomials[0].coeff <= c.scalar :
                    print(x)
                    return True
            elif type == "<":
                if x * c.monomials[0].coeff >= c.scalar :
                    print(x)
                    return True
    return False

# create controller object with a keras model
# good controller
# model = load_model("../OverApprox/models/single_pend_nn_controller_ilqr_data.h5")
model = load_model("../OverApprox/models/single_pend_nn_controller_lqr_data.h5")
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

single_pendulum_dynamics = Dynamics(None, np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
single_pendulum_dynamics.constraints = overt_obj.constraints.copy()
next_states = single_pendulum_dynamics.next_states.reshape(2,)

# x_next = x + dt*u
dt = 0.1
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, states[0]), Monomial(dt, states[1]), Monomial(-1, next_states[0])]

# u_next = u + dt*a
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1, states[1]), Monomial(dt, acceleration), Monomial(-1, next_states[1])]

single_pendulum_dynamics.constraints += [c1, c2]

print("single pendulum dynamics constraints = ", len(single_pendulum_dynamics.constraints))
print("controler constraints = ", len(controller.constraints))

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=single_pendulum_dynamics,
                                        controller_obj=controller)

# initial set
x1_init_set = (0.6, 0.8)
x2_init_set = (-1., 1.)
init_set = {states[0]: x1_init_set, states[1]: x2_init_set}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

prop_list =[]
p1 = Constraint(ConstraintType('GREATER'))
p1.monomials = [Monomial(1, states[0])]
p1.scalar = 0.2
prop_list.append(p1)

p2 = Constraint(ConstraintType('LESS'))
p2.monomials = [Monomial(1, states[0])]
p2.scalar = 1.08
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
violation_found = False
n_simulation = 10000
for i in range(n_simulation):
    th = np.random.uniform(init_set[states[0]][0], init_set[states[0]][1])
    dth = np.random.uniform(init_set[states[1]][0], init_set[states[1]][1])
    for j in range(ncheck_invariant-1):
        T = model.predict(np.array([th, dth]).reshape(1,2))[0][0]
        th, dth = single_pendulum(th, dth, T, dt)
        if th < p1.scalar or th > p2.scalar:
            print("violation was found: %0.3f" %th)
            violation_found = True
            break
    if violation_found:
        break

if not violation_found:
    print("No violation was found in %d simulations" %n_simulation)




# n_repeat = 10000
# no_vi = True
# for _ in range(n_repeat):
#     x_0_0 = np.random.uniform(init_set[states[0]][0], init_set[states[0]][1])
#     x_0_1 = np.random.uniform(init_set[states[1]][0], init_set[states[1]][1])
#     x_0 = [x_0_0, x_0_1]
#     env = Pendulum1Env(x_0=x_0, dt=dt)
#     env.reset()
#     for time in range(ncheck_invariant-1):
#         # print("time: %d, th=%0.3f, thdot=%0.3f" %(time, env.x[0], env.x[1]))
#         torque = model.predict(env.x.reshape(-1,2)).reshape(1)
#         env.step(torque)
#
#     if property_violated(env, prop):
#         print("***property was violated***")
#         no_vi = False
#         break
#
# if no_vi:
#     print("no violation found in %d simulations" %n_repeat)


# env.render()

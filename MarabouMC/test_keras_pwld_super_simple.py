
import os
import sys
import random
import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial, MaxConstraint, ReluConstraint
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC

# the goal here is to test a very simple dynamical simple with a very simple control.
#              dx = -1/2 + c
# where c is the control parameter. We take
#              c = alpha*x
# If we discretize this with dt=1, we get
#              x_{n+1} - x_n = -1/2 + alpha*x_n
# which simplifies to
#              x_{n+1} = -1/2 + x_n(alpha + 1)

#        |  alpha = -1 | alpha = -1/2  |   alpha = 0  |  alpha = 1/2 |   alpha = 1   |
# =======|=============|===============|==============|==============|===============|
#   x0   |    [0,1]    |    [0,1]      |    [0,1]     |    [0,1]     |     [0,1]     |
#   x1   |    -1/2     |    [-1,0]     |  [-1/2,1/2]  |   [-1/2,1]   |   [-1/2,3/2]  |
#   x2   |    -1/2     |    [-1,0]     |    [-1,0]    |   [-5/4,1]   |   [-3/2,5/2]  |
#   x3   |    -1/2     |    [-1,0]     |  [-3/2,-1/2] |   [-19/8,1]  |   [-7/2,9/2]  |
#   x4   |    -1/2     |    [-1,0]     |    [-2,-1]   |  [-65/16,1]  |  [-15/2,17/2] |
# create controller object, this is just a place holder. I will modify the object later.


model = load_model("../OverApprox/models/single_pend_nn_controller_lqr_data.h5")
controller = KerasController(keras_model=model)

# rewrite to make a simple controller that is always equal to alpha*x
alpha = -1/2
controller.control_outputs = [['c']]
controller.state_inputs = [['xc']]
controller.relus = []
monomial_list = [Monomial(alpha, controller.control_outputs[0][0]), Monomial(-1, controller.state_inputs[0][0])]
fake_constraint = [Constraint(ConstraintType('EQUALITY'), monomial_list, 0.0)]
controller.constraints = fake_constraint


# create overt dynamics objects. this is just a place holder. I will modify the object later.
overt_obj = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")

# rewrite to make a simple controller that is always equal to x
overt_obj.control_vars = [['cd']]
overt_obj.state_vars = [['x']]
overt_obj.output_vars = [['dx']]
monomial_list2 = [Monomial(1, overt_obj.control_vars[0][0]), Monomial(-1, overt_obj.output_vars[0][0])]
fake_constraint2 = [Constraint(ConstraintType('EQUALITY'), monomial_list2, 0.5)]
overt_obj.constraints = fake_constraint2

simple_dynamics = Dynamics(None, np.array(overt_obj.state_vars).reshape(-1, 1), np.array(controller.control_outputs).reshape(-1, 1))
next_states = simple_dynamics.next_states.reshape(1,)

# x_next = x + dt*dx
dt = 1
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, overt_obj.state_vars[0][0]), Monomial(dt, overt_obj.output_vars[0][0]), Monomial(-1, next_states[0])]

simple_dynamics.constraints = [c1] + overt_obj.constraints

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=simple_dynamics,
                                        controller_obj=controller)

# initial set
init_set = {overt_obj.state_vars[0][0]: (0., 1.)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

# property x< 0.105, x' < 0.2
p = Constraint(ConstraintType('GREATER'))
p.monomials = [Monomial(1, overt_obj.state_vars[0][0])]
p.scalar = -1/4 #
prop = ConstraintProperty([p])


# p.monomials = [Monomial(1, states[0])]
# p.scalar = 0.101 # 0 #
# prop = ConstraintProperty([p])

# algo
algo = BMC(ts = ts, prop = prop, solver=solver)
algo.check_invariant_until(5)

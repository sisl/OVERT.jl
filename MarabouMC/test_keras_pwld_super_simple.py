
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

# test one, c = alpha*x.
#        |  alpha = -1 | alpha = -1/2  |   alpha = 0  |  alpha = 1/2 |   alpha = 1   |
# =======|=============|===============|==============|==============|===============|
#   x0   |    [0,1]    |    [0,1]      |    [0,1]     |    [0,1]     |     [0,1]     |
#   x1   |    -1/2     |   [-1/2,0]    |  [-1/2,1/2]  |   [-1/2,1]   |   [-1/2,3/2]  |
#   x2   |    -1/2     |  [-3/4,-1/2]  |    [-1,0]    |   [-5/4,1]   |   [-3/2,5/2]  |
#   x3   |    -1/2     |  [-7/8,-3/4]  |  [-3/2,-1/2] |   [-19/8,1]  |   [-7/2,9/2]  |
#   x4   |    -1/2     | [-15/16,-7/8] |    [-2,-1]   |  [-65/16,1]  |  [-15/2,17/2] |

# test two, now let c = Relu(alpha*x)
#        |  alpha = -1 | alpha = -1/2  |   alpha = 0  |  alpha = 1/2 |   alpha = 1   |
# =======|=============|===============|==============|==============|===============|
#   x0   |    [0,1]    |    [0,1]      |    [0,1]     |    [0,1]     |     [0,1]     |
#   x1   | [-1/2,1/2]  |  [-1/2,1/2]   |  [-1/2,1/2]  |   [-1/2,1]   |   [-1/2,3/2]  |
#   x2   |  [-1/2,0]  |   [-3/4,0]    |     [-1,0]   |    [-1,1]    |    [-1,5/2]   |
#   x3   |    -1/2     |  [-7/8,-1/2]  |  [-3/2,-1/2] |   [-3/2,1]   |   [-3/2,9/2]  |
#   x4   |    -1/2     | [-15/16,-3/4] |    [-2,-1]   |    [-2,1]    |    [-2,17/2]  |

# test three, now let c = max(alpha*x, -1/2)
#        |  alpha = -1 | alpha = -1/2  |   alpha = 0  |  alpha = 1/2 |   alpha = 1   |
# =======|=============|===============|==============|==============|===============|
#   x0   |    [0,1]    |     [0,1]     |    [0,1]     |    [0,1]     |     [0,1]     |
#   x1   |   [-1/2,0]  |    [-1/2,0]   |  [-1/2,1/2]  |   [-1/2,1]   |    [-1/2,3/2] |
#   x2   |    -1/2     |  [-3/4,-1/2]  |     [-1,0]   |    [-5/4,1]  |   [-3/2,5/2]  |
#   x3   |    -1/2     |  [-7/8,-3/4]  |  [-3/2,-1/2] |   [-9/4,1]   |   [-5/2,9/2]  |
#   x4   |    -1/2     | [-15/16,-7/8] |    [-2,-1]   |   [-13/4,1]  |   [-7/2,17/2] |


def test_marabou_interface(alpha, prop_desc, n_invar, with_relu=False, with_max=False):
    # create controller object, this is just a place holder. I will modify the object later.
    model = load_model("../OverApprox/models/single_pend_nn_controller_lqr_data.h5")
    controller = KerasController(keras_model=model)

    # rewrite to make a simple controller that is always equal to alpha*x
    controller.control_outputs = [['c']]
    controller.state_inputs = [['xc']]
    fake_constraint = []
    if with_relu:
        alpha_times_x = 'var1'
        monomial_list = [Monomial(alpha, controller.state_inputs[0][0]), Monomial(-1, alpha_times_x)]
        fake_constraint.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0.0))
        relu_constraint = [ReluConstraint(varin=alpha_times_x, varout=controller.control_outputs[0][0])]
        controller.constraints = relu_constraint + fake_constraint
        controller.relus = relu_constraint
    elif with_max:
        alpha_times_x = 'var1'
        monomial_list = [Monomial(alpha, controller.state_inputs[0][0]), Monomial(-1, alpha_times_x)]
        fake_constraint.append(Constraint(ConstraintType('EQUALITY'), monomial_list, 0.0))
        max_second_arg = 'var2'
        fake_constraint.append(Constraint(ConstraintType('EQUALITY'), [Monomial(1, max_second_arg)], -1/2))
        max_constraint = [MaxConstraint(varsin=[alpha_times_x, max_second_arg], varout=controller.control_outputs[0][0])]
        controller.constraints = max_constraint + fake_constraint
        controller.relus = []
    else:
        monomial_list = [Monomial(-1, controller.control_outputs[0][0]), Monomial(alpha, controller.state_inputs[0][0])]
        fake_constraint = [Constraint(ConstraintType('EQUALITY'), monomial_list, 0.0)]
        controller.constraints = fake_constraint
        controller.relus = []


    # create overt dynamics objects. this is just a place holder. I will modify the object later.
    overt_obj = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")

    # rewrite to make a simple controller that is always equal to x
    overt_obj.control_vars = [['cd']]
    overt_obj.state_vars = [['x']]
    overt_obj.output_vars = [['dx']]
    monomial_list2 = [Monomial(1, overt_obj.control_vars[0][0]), Monomial(-1, overt_obj.output_vars[0][0])]
    fake_constraint2 = [Constraint(ConstraintType('EQUALITY'), monomial_list2, 0.5)]
    overt_obj.constraints = fake_constraint2

    simple_dynamics = Dynamics(None, np.array(['x']), np.array(['cd']))
    next_states = simple_dynamics.next_states.reshape(1,)

    # x_next = x + dt*dx
    dt = 1
    c1 = Constraint(ConstraintType('EQUALITY'))
    c1.monomials = [Monomial(1, overt_obj.state_vars[0][0]), Monomial(dt, overt_obj.output_vars[0][0]), Monomial(-1, next_states[0])]

    simple_dynamics.constraints = [c1] + overt_obj.constraints

    print(len(simple_dynamics.constraints))
    print(len(controller.constraints))


    # create transition relation using controller and dynamics
    tr = TFControlledTransitionRelation(dynamics_obj=simple_dynamics,
                                            controller_obj=controller)

    # initial set
    init_set = {overt_obj.state_vars[0][0]: (0., 1.)}

    # build the transition system as an (S, I(S), TR) tuple
    ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

    # property x< 0.105, x' < 0.2
    p = Constraint(ConstraintType(prop_desc["type"]))
    p.monomials = [Monomial(1, overt_obj.state_vars[0][0])]
    p.scalar = prop_desc["scalar"]  #
    prop = ConstraintProperty([p])

    # solver
    solver = MarabouWrapper()
    algo = BMC(ts = ts, prop = prop, solver=solver)
    result = algo.check_invariant_until(n_invar)
    return result.name

#print(test_marabou_interface(-1, {"type": "GREATER", "scalar":-3.}, 4) == "UNSAT")
#print(test_marabou_interface(-1/2, {"type": "GREATER", "scalar":-7/8}, 2, with_relu=True) == "UNSAT")
#print(test_marabou_interface(-1/2, {"type": "GREATER", "scalar":-6.9/8}, 4, with_relu=True) == "UNSAT")
#print(test_marabou_interface(-1/2, {"type": "GREATER", "scalar":-7/8}, 4, with_relu=True) == "UNSAT")
print(test_marabou_interface(-1/2, {"type": "GREATER", "scalar":-7.1/8}, 5, with_max=True) == "UNSAT")
#print(test_marabou_interface(1, {"type": "GREATER", "scalar":-3.}, 5, with_max=True) == "UNSAT")
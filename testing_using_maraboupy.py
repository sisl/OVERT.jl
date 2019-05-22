# testing stuff . taken out of using_maraboupy
from maraboupy import Marabou
import numpy as np
import os
from set_pendulum_bounds import *
from bounds_funs import *
from using_marabou_utils import *
import sys

# test TF eval
testTF = False
if testTF:
    filename = frozen_graph
    theta = 0.067725644048100
    theta_dot = -0.218659060007169
    input_dict = {"initial_values/theta_0": [[theta]],
                  "initial_values/theta_dot_0": [[theta_dot]], 
                    }
    print("intitial theta: ", theta*180/np.pi)
    print("initial thetadot: ", theta_dot*180/np.pi)
    outputs = ["get_action/nonlinearity_2/output",
               "run_dynamics/Dynamics_1/theta_1",
               "run_dynamics/Dynamics_1/thdotLB/theta_dot_LB_1",
               "run_dynamics/Dynamics_1/thdotUB/theta_dot_UB_1"
    ]
    outputs = eval_from_tf(filename, input_dict, outputs)
    print("outputs from tf: ", outputs)

# eval and make sure they produce the same outputs
run = False
if run:
    network.setLowerBound(2, -1000.)
    network.setUpperBound(2, 1000.)
    network.setLowerBound(3, -1000.)
    network.setUpperBound(3, 1000.)
    theta = 90*np.pi/180
    theta_dot = -0.01*np.pi/180
    compare_marabou_tf(network, theta, theta_dot, nsteps, outputVarList)

evalPoint = True 
if evalPoint:
    theta = 0.020675993904395
    thetadot = 0.940993002874093
    vals, stats, exit_code = eval_marabou(network, theta, thetadot)

# plot = True
# if plot:
#     filename = frozen_graph
#     fun = lambda theta : u/(m*l*l) + (g/l)*np.sin(theta)
#     fixed_vars = {

#     }
#     var_to_sample = {"initial_values/theta_0" : [-np.pi, np.pi]}
#     plot_bounded_dynamics(filename, fixed_vars, var_to_sample, outputs, fun)

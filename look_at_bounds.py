from using_marabou_utils import  plot_bounded_dynamics
import numpy as np

# look at whole graph
# plot a bunch of things.
# e.g. first plot dynamics + bound alone. control = 0
# then plot output of several steps of whole network

filename = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files/"
filename += "graph_def_44693.pb"

var_to_sample = ["initial_values/theta_0", [-90*np.pi/180, 90*np.pi/180]]#["relu_protection/theta_1", [-90*np.pi/180, 90*np.pi/180]]
outputs = ["relu_protection/matmul",
           "run_dynamics/Dynamics_1/thdotLB/additive_bound/add",
           "run_dynamics/Dynamics_1/thdotUB/additive_bound/add"]
def fun(th):
    return (9.8/0.1)*np.sin(th)
fixed_vars = {"initial_values/theta_dot_0:0": [[0]]}
plot_bounded_dynamics(filename, fixed_vars, var_to_sample, outputs, fun)
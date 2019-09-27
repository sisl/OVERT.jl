# top level user script to solve a multi-timestep marabou user query one timestep at a time
from one_step_at_a_time_utils import solve_by_timestep
import sys
from using_marabou_utils import load_network_wrapper, set_up_logging
from bounds_funs import *

# load concatenated graph
fnumber = "38358"
fname = "graph_def_" #real_controller_2_steps_"
nsteps = 4 # only used in lookin at specific equations and overapprox checking
fprefix = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files"
frozen_graph, output_op_name, inputs = load_network_wrapper(fprefix, fnumber, fname)

# some more logging
logname, marabou_log_dir, network_dir, run_number = set_up_logging(fnumber)

# solve one timestep at a time
found_SAT = solve_by_timestep(frozen_graph, output_op_name, inputs, bounds_4_5, network_dir, run_number, nsteps, marabou_log_dir, logname)

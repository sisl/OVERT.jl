# script to parse .pb into marabou
from maraboupy import Marabou
import numpy as np
import os
from set_pendulum_bounds import *
from bounds_funs import *
from using_marabou_utils import *
import sys
import gym
from simple_wrapper import SimpleWrapper

true_stdout = sys.stdout

# graph_def_gentler_random_controller_2_steps_4856.pb
fnumber = "38358"
fname = "graph_def_" #real_controller_2_steps_"
nsteps = 4 # only used in lookin at specific equations and overapprox checking
fprefix = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files"

# load network
frozen_graph, output_op_name, inputs = load_network_wrapper(fprefix, fnumber, fname)
network = Marabou.read_tf(frozen_graph, outputName=output_op_name)

# set up logging
logname, marabou_log_dir, network_dir, run_number = set_up_logging(fnumber)
sys.stdout = open(logname, 'w')
# some debugging
inputVars = network.inputVars
print("inputVars:", inputVars)
outputVars = network.outputVars
print("outputVars: ", outputVars)
outputVarList = list(np.array(outputVars).flatten())

# set bounds
#d1, d2 = map_inputs(network, inputs) # for use with controllers parsed using parsing.py into ff networks

d1, d2 = map_inputs_fromVarMap(varMapOpstoNames(network.varMap), inputs) # for use with other networks that have not been condensed

# set bounds on outputs
# TODO: want there to be an error if number of steps in bounds don't match number of steps of network
bounds = set_bounds(network, d1, d2, bounds_4_5, network_dir, run_number)

# make sure all lower bounds are less than all upper bounds
check_bounds(network.upperBounds, network.lowerBounds)

# check upper and lower bounds of inputs
print_io_bounds(network, inputVars, outputVars)

# debugging
equns3 = find_spec_equ(network.equList, outputVars[:nsteps]) #
[print(e) for e in equns3]

# run_script('testing_using_maraboupy.py')

solve=True
if solve:
    sys.stdout = true_stdout
    vals, stats, exit_code = solve_with_marabou(network, marabou_log_dir)
    sys.stdout = open(logname, 'a')
    print("exit code: ", exit_code)
    if exit_code == 1: # SAT    
        SATus = check_SAT(frozen_graph, vals, bounds, nsteps)
    sys.stdout = true_stdout

"""
recall:
enum ExitCode {
        UNSAT = 0,
        SAT = 1,
        ERROR = 2,
        TIMEOUT = 3,

        NOT_DONE = 999,
    };
"""







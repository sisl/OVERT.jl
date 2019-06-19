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
run_number = str(int(np.ceil(np.random.rand()*10000)))
fnumber = "41475"
fname = "graph_def_" #real_controller_2_steps_"
nsteps = 2 # only used in lookin at specific equations and overapprox checking
fprefix = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files"
frozen_graph = os.path.join(fprefix, fname+fnumber+".pb")
meta_data = os.path.join(fprefix, "meta_data_"+fnumber+".txt")

# make path in which to store outputs
network_dir = '/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/MarabouLogs/network_'+fnumber
if not os.path.exists(network_dir):
    os.mkdir(network_dir)

marabou_log_dir = os.path.join(network_dir, 'run_'+run_number+'_marabou.log')
print(marabou_log_dir)
if os.path.exists(marabou_log_dir): # don't overwrite old data!!!
    raise FileExistsError

output_op_name, inputs, outputs = read_inout_metadata(meta_data)

network = Marabou.read_tf(frozen_graph, outputName=output_op_name)

# redirect to file
logname = os.path.join(network_dir,'run_'+run_number+'_peripheral.log')
sys.stdout = open(logname, 'w')

inputVars = network.inputVars
print("inputVars:", inputVars)
outputVars = network.outputVars
print("outputVars: ", outputVars)
outputVarList = list(np.array(outputVars).flatten())

#d1, d2 = map_inputs(network, inputs) # for use with controllers parsed using parsing.py into ff networks

d1, d2 = map_inputs_fromVarMap(varMapOpstoNames(network.varMap), inputs) # for use with other networks that have not been condensed

# set bounds on outputs
bounds = set_bounds(network, d1, d2, bounds_3_5, network_dir, run_number)

# make sure all lower bounds are less than all upper bounds
check_bounds(network.upperBounds, network.lowerBounds)

# check upper and lower bounds of inputs
print_io_bounds(network, inputVars, outputVars)

equns3 = find_spec_equ(network.equList, outputVars[:nsteps]) #
[print(e) for e in equns3]

# run_script('testing_using_maraboupy.py')

# call Marabou solver
def solve_with_marabou(network, marabou_log_dir):
    vals, stats, exit_code = network.solve(marabou_log_dir)
    if len(vals)>0:
        print("Input vals: ", [vals[iv] for iv in np.array(inputVars).flatten()])
        print("Output vals: ", [vals[iv] for iv in np.array(outputVars).flatten()])
    print("stats: ", stats)
    return vals, stats, exit_code

solve=True
if solve:
    sys.stdout = true_stdout
    print("solving...")
    vals, stats, exit_code = solve_with_marabou(network, marabou_log_dir)
    sys.stdout = open(logname, 'a')
    print("exit code: ", exit_code)
    if exit_code == 1: # SAT
        print(vals)
        env = SimpleWrapper(gym.envs.make('MyPendulum-v0')) # , recordVideo=True)
        sess = load_network(frozen_graph)
        SATus = check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps)
        print("SATus:", SATus)
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







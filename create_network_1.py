# script to create MarabouNetwork object to use with DnC solver
from maraboupy import Marabou
import numpy as np
import os
from set_pendulum_bounds import *
from bounds_funs import *
from using_marabou_utils import *
import sys

def create_network():

    true_stdout = sys.stdout

    run_number = str(int(np.ceil(np.random.rand()*10000)))
    fnumber = "3332"
    fname = "graph_def_" #real_controller_2_steps_"
    nsteps = 4
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

    # redirect to file
    peripheral_logfile = os.path.join(network_dir,'run_'+run_number+'_peripheral.log')
    sys.stdout = open(peripheral_logfile, 'w')

    output_op_name, inputs, outputs = read_inout_metadata(meta_data)

    network = Marabou.read_tf(frozen_graph, outputName=output_op_name)
    network.name = frozen_graph[0:-3] # strip off .pb at the end

    inputVars = network.inputVars
    print("inputVars:", inputVars)
    outputVars = network.outputVars
    print("outputVars: ", outputVars)
    outputVarList = list(np.array(outputVars).flatten())

    d1, d2 = map_inputs_fromVarMap(varMapOpstoNames(network.varMap), inputs) # for use with other networks that have not been condensed

    # set adjustable vars for DnC
    network.adjustable_inputs = get_adjustable_vars(["initial_values/theta_0","initial_values/theta_dot_0"], d2)

    network.dependencies = get_dependencies(nsteps)

    # set bounds on outputs
    bounds = set_bounds(network, d1, d2, bounds_2_5, network_dir, run_number)
    network.bounds = bounds
    network.nsteps = nsteps

    # make sure all lower bounds are less than all upper bounds
    check_bounds(network.upperBounds, network.lowerBounds)

    sys.stdout = true_stdout
    network.peripheral_logfile = peripheral_logfile

    return network, peripheral_logfile, marabou_log_dir

def SAT_callback(logfile, SATvals, network):
    true_stdout = sys.stdout
    sys.stdout = open(logfile, 'a')
    print("Evaluating SATus...")
    envStr = 'MyPendulum-v0'
    SATus = check_SAT_REAL_or_OVERAPPROX(network.name+".pb", SATvals, envStr, network.bounds, network.nsteps, render=False)
    print("SATus:", SATus)
    sys.stdout = true_stdout



















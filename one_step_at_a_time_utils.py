# utils for solving a multi-timestep marabou user query one timestep at a time
from set_pendulum_bounds import set_k_bounds
from using_marabou_utils import *

# solve one timestep at a time function
def solve_by_timestep(network, il2s, il2v, bounds_fun, fdir, run_n, nsteps):
	found_SAT = False
	step = 1
	while (not found_SAT) and (step <= nsteps):
		# set bounds on graph
		set_k_bounds(network, il2s, il2v, bounds_fun, fdir, run_n, step)
		#####################

		# check and log bounds	
		# make sure all lower bounds are less than all upper bounds
		check_bounds(network.upperBounds, network.lowerBounds)
		# check upper and lower bounds of inputs
		print_io_bounds(network, inputVars, outputVars)
		#####################

		# solve with marabou
		vals, stats, exit_code = solve_with_marabou(network, marabou_log_dir)
    	print("exit code: ", exit_code)
	    if exit_code == 1: # SAT   
	    	print("Found SAT: ", found_SAT, " at tstep ", step) 
	        SATus = check_SAT(frozen_graph, vals, bounds, nsteps)
	        found_SAT = True
	    #####################
		step += 1

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
	return exit_code
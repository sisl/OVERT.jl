# utils for solving a multi-timestep marabou user query one timestep at a time
from set_pendulum_bounds import set_k_bounds, map_inputs_fromVarMap
from using_marabou_utils import *

# solve one timestep at a time function
def solve_by_timestep(frozen_graph, output_op_name, inputs, bounds_fun, network_dir, run_n, nsteps, marabou_log_dir, logname):
	# logging
	true_stdout = sys.stdout
	sys.stdout = open(logname, 'w')
	network = Marabou.read_tf(frozen_graph, outputName=output_op_name)
	d1, d2 = map_inputs_fromVarMap(varMapOpstoNames(network.varMap), inputs) # for use with other networks that have not been condensed
	found_SAT = False
	step = 1
	while (not found_SAT) and (step <= nsteps):
		# set bounds on graph
		bounds = set_k_bounds(network, d1, d2, bounds_fun, network_dir, run_n, step)
		#####################

		# check and log bounds	
		# make sure all lower bounds are less than all upper bounds
		check_bounds(network.upperBounds, network.lowerBounds)
		# check upper and lower bounds of inputs
		print_io_bounds(network, network.inputVars, network.outputVars)
		#####################

		# solve with marabou
		sys.stdout = true_stdout
		vals, stats, exit_code = solve_with_marabou(network, marabou_log_dir)
		sys.stdout = open(logname, 'a')
		print("step: ", step)
		print("exit code: ", exit_code)
		if exit_code == 1: # SAT
			print("Found SAT: ", found_SAT, " at tstep ", step)
			SATus = check_SAT(frozen_graph, vals, bounds, nsteps)
			found_SAT = True
		sys.stdout = true_stdout
		#####################
		step += 1
		# re-load network for safety
		network = Marabou.read_tf(frozen_graph, outputName=output_op_name)

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
	return found_SAT
# top level user script to solve a multi-timestep marabou user query one timestep at a time
from one_step_at_a_time_utils import solve_by_timestep
import sys

# set up logging

# load concatenated graph

# some more logging

# solve one timestep at a time
solve_by_timestep(network, il2s, il2v, bounds_fun, fdir, run_n, nsteps)
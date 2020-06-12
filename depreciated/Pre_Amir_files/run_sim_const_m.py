# run simulation using simulation.py
# assumes varying mass

from simulation import sim_system
from sandbox.rocky.tf.envs.vary_wrapper import VaryMassRolloutWrapper
import os
import numpy as np
import tensorflow as tf

prefix = "/Users/Chelsea/Dropbox/AAHAA/src/rllab/data/local/experiment/"
policy_name = "EXCELLENT_POLICY_relu_small_network_ppo_capped_action_simpler_dense_layer_xW_learn_std_smaller_learning_rate"
policy_file = os.path.join(prefix, policy_name, 'params.pkl')
start_range = ([-5*np.pi/180, -60*np.pi/180],
			   [5*np.pi/180, 60*np.pi/180])
#([-10*np.pi/180, -120*np.pi/180],[10*np.pi/180, 120*np.pi/180])

safe_range = ([-10*np.pi/180, -60*np.pi/180],[10*np.pi/180, 60*np.pi/180])
#([-0.17453292519943295, -2.0943951023931953],
#			  [ 0.17453292519943295,  2.0943951023931953]
#			 )
nsteps = 2
nsims = 150
wrap_fn = None

# labels for plotting are in deg but computations happen in radians
labels = ["theta, deg", "thetadot, deg/s"]

with tf.Session() as sess:
	sim_system(policy_file, policy_name, start_range, safe_range, nsteps, nsims, wrap_fn, labels=labels, plot_trajs=True, rand=False)


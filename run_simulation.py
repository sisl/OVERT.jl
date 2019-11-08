# run simulation using simulation.py
# assumes varying mass

from simulation import sim_system
from sandbox.rocky.tf.envs.vary_wrapper import VaryMassRolloutWrapper
import os
import numpy as np
import tensorflow as tf

net_n = "44478"
prefix = "/Users/Chelsea/Dropbox/AAHAA/src/Overapprox/nnet_files/"
file = open(os.path.join(prefix, "policy_file_data"+net_n+".txt"), "r").readline()[:-1] # cut off \n character
policy_name = file.split('/')[-2]
print("using policy: ", policy_name)
start_range = ([-5*np.pi/180, -120*np.pi/180],
			   [10*np.pi/180, 120*np.pi/180])
#([-10*np.pi/180, -120*np.pi/180],[10*np.pi/180, 120*np.pi/180])

safe_range = ([-5*np.pi/180, -120*np.pi/180],[10*np.pi/180, 120*np.pi/180])
#([-0.17453292519943295, -2.0943951023931953],
#			  [ 0.17453292519943295,  2.0943951023931953]
#			 )
nsteps = 3
nsims = 150
wrap_fn = VaryMassRolloutWrapper

# labels for plotting are in deg but computations happen in radians
labels = ["theta, deg", "thetadot, deg/s"]

with tf.Session() as sess:
	sim_system(file, policy_name, start_range, safe_range, nsteps, nsims, wrap_fn, labels=labels, plot_trajs=True, rand=True)


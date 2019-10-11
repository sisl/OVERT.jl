# sim script to validate UNSAT results. Could be used with safety proofs that 
# start from a single point and never exceed a safety box, or safety proofs that
# start in some range and never exceed some largest "epsilon Lyapunov set"

from rllab.sampler.utils import get_deterministic_action, deterministic_rollout
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sandbox.rocky.tf.envs.vary_wrapper import VaryMassRolloutWrapper

# function: take in network name, env, starting range, safe range, steps per sim, 
#           number of sims
def sim_system(policy_file, pol_name, start_range, safe_range, nsteps, nsims, env_wrap_fn=None, labels=[""], plot_trajs=True, rand=True):
	# start_range: ([low], [high]) tuple of two array-like representing an 
	#                              interval
	# 
	# note: Intended for use with 1D observations

	# make a directory to hold sim data for this run of this policy
	sim_ID, dirname = make_dir(pol_name)

	# load control network AND env
	policy, env = load(policy_file, env_wrap_fn)

	env.reset()
	if env_wrap_fn is not None:
		env.env._wrapped_env.env.enabled = False
	else:
		env._wrapped_env.env.enabled = False

	if not rand:
		n = np.ceil(np.sqrt(nsims))
		out = np.meshgrid(np.linspace(start=start_range[0][0], stop=start_range[1][0], num=n),
						np.linspace(start=start_range[0][1], stop=start_range[1][1], num=n))
		ICs = np.vstack([out[0].flatten(), out[1].flatten()])
		nsims = ICs.shape[1]

	# storage containers
	trajs = np.ones((env.observation_space.flat_dim, nsteps + 1, nsims)) * 55
	mass = np.ones((nsims)) * 55

	# for some number of sims
	for i in range(nsims):
		# generate IC in starting set
		if rand:
			IC = np.random.uniform(low=start_range[0],
							   high=start_range[1], 
							   size=np.array(start_range[0]).flatten().shape)
		else:
			IC = ICs[:,i].flatten()

		# simulate for some number of steps for sim
		# do stuff with mass if mass varies
		if env_wrap_fn is VaryMassRolloutWrapper:
			if rand:
				env.set_mass_randomly()
			else:
				env.env.set_param('m', 0.25) #env.env.mf)
			mass[i] = env.env.get_param('m')
		elif env_wrap_fn is None:
			mass[i] = env._wrapped_env.env.env.env.m
		print("mass: ", mass[i])
		# SET INITIAL CONDITION
		if env_wrap_fn is VaryMassRolloutWrapper:
			env.set_state(IC) # SET INITIAL CONDITION
			trajs[:, 0, i] = env.get_state()
		elif env_wrap_fn is None:
			env._wrapped_env.env.env.env.state = IC
			trajs[:, 0, i] = env._wrapped_env.env.env.env.state
		s = trajs[:, 0, i]
		for j in range(nsteps):
			# get control action
			a, info = get_deterministic_action(policy, s)
			# step environment
			s, r, d, info = env.step(a)
			# record state trajectories
			trajs[:,j+1,i] = s

	# convert to degrees and deg/sec
	trajs = trajs*180/np.pi

	# plot safe set. Assume 2D obs space
	fig = plt.figure()
	# plot trajectories
	for i in range(nsims):
		if i == 0:
			plt.plot(trajs[0,0,i], trajs[1,0,i], 'x', markersize=10, label="start")
		else:
			plt.plot(trajs[0,0,i], trajs[1,0,i], 'x', markersize=10)
		if i == 0:
			plt.plot(trajs[0,-1,i], trajs[1,-1,i], 'o', markersize=10, label="end")
		else:
			plt.plot(trajs[0,-1,i], trajs[1,-1,i], 'o', markersize=10)

	# label extreme mass ending points if starting from point
	if np.linalg.norm(np.array(start_range[0]) - np.array(start_range[1])) < 1e-3:
		i_min = np.argmin(mass)
		i_max = np.argmax(mass)
		plt.plot(trajs[0,-1,i_min], trajs[1,-1,i_min], 'k*', markersize=10, label="%.2f kg"%mass[i_min] )
		plt.plot(trajs[0,-1,i_max], trajs[1,-1,i_max], 'c*', markersize=10, label="%.2f kg"%mass[i_max] )

	plt.plot([0], [0], '*', label="origin")

	plt.legend()
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title("Simulation Trajectories")
	plt.savefig(os.path.join(dirname, "plot_1.pdf"))

	safe_range = convert_to_deg(safe_range)
	start_range = convert_to_deg(start_range)
	x,y = get_box_pts(safe_range[0],safe_range[1])
	plt.plot(x, y, label="safe set")

	# plot init set
	x,y = get_box_pts(start_range[0], start_range[1])
	plt.plot(x,y, 'x-', markersize=10, label="init set")
	plt.legend()

	# save trajectory data
	np.save(os.path.join(dirname, "trajs.npy"), trajs)
	np.save(os.path.join(dirname, "masses.npy"), mass)

	# save ranges
	pickle.dump(safe_range, open(os.path.join(dirname, "safe_set.pkl"), "wb"))
	pickle.dump(start_range, open(os.path.join(dirname, "start_set.pkl"), "wb"))

	# test. this is how to load the data
	# data = pickle.load(open(os.path.join(dirname, "safe_set.pkl"), "rb"))

	# save plot
	plt.savefig(os.path.join(dirname, "plot_2.pdf"))

	# actually plot trajectories
	if plot_trajs:
		for i in range(nsims):
			plt.plot(trajs[0,:,i], trajs[1,:,i], '.-')
		plt.savefig(os.path.join(dirname, "plot_3.pdf"))

	plt.show()

def make_dir(pol_name):
	sim_ID = str(int(np.ceil(np.random.rand()*50000)))
	dirname = "simulation_logs/"+pol_name
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	runname = dirname+"/verification_sim_"+sim_ID
	if not os.path.exists(runname):
		os.mkdir(runname)
	else:
		raise FileExistsError
	return sim_ID, runname

def reset(env, IC):
	env.set_mass_randomly()
	env.set_state(IC)

def convert_to_deg(r):
	# convert intervals to degrees
	a = np.array(r[0])
	b = np.array(r[1])
	return (a*180/np.pi, b*180/np.pi)

def get_box_pts(L,H):
	x = [L[0], H[0], H[0], L[0], L[0]]
	y = [L[1], L[1], H[1], H[1], L[1]]
	return x, y

def load(file, env_wrap_fn=None, test=False):
	data = joblib.load(file)
	policy = data['policy']
	if env_wrap_fn is None:
		env = data['env']
	else:
		env = env_wrap_fn(data['env'])

	if test:
		deterministic_rollout(env, policy, max_path_length=200,
                       animated=True, speedup=100, always_return_paths=False)

	return policy, env

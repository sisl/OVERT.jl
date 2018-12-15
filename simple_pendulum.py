# simple pendulum problem

# theta_dd = F/ml  - gsin(theta)/l
# theta_dd = Tau/(ml^2) - (g/l)*sin(theta)

import tensorflow as tf 
import numpy as np
from simple_overapprox_simple_pendulum import line, bound, build_sin_approx
import os
import joblib
from rllab.sampler.utils import rollout
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

# inputs: state and action yields next state
def create_dynamics_block(num, var_dict):
	with tf.name_scope("Dynamics_"+str(num)):
		torque_UB = var_dict["action_UB"]
		theta_t_UB = var_dict["theta_t_UB"]
		theta_d_t_UB = var_dict["theta_d_t_UB"]
		torque_LB = var_dict["action_LB"]
		theta_t_LB = var_dict["theta_t_LB"]
		theta_d_t_LB = var_dict["theta_d_t_LB"]
		#with tf.variable_scope("time_"+str(num)):
		#	torque_UB = tf.Variable([var_dict["torque_UB"]], name="Torque_UB")
		#	torque_LB = tf.Variable([0.0], name="Torque_LB")
		#	theta_t_UB = tf.Variable([1.0], name="theta_t_UB")
		#	theta_t_LB = tf.Variable([1.0], name="theta_t_LB")
		#	theta_d_t_UB = tf.Variable([0.0], name="theta_d_t_UB")
		#	theta_d_t_LB = tf.Variable([0.0], name="theta_d_t_LB")

		m = 0.25 # kg
		l = 0.1 # m
		g = 9.8 # m/s^2
		oomls = tf.constant([(1/ (m*(l**2)) )], name="torque_scaling")

		c1 = g/l
		c2_UB = torque_UB*oomls
		c2_LB = torque_LB*oomls

		fun = lambda x: c1*np.sin(x)

		# build_sin_approx(fun, c1, convex_reg, concave_reg)
		LB, UB = build_sin_approx(fun, c1, [-np.pi, 0], [0, np.pi])

		with tf.name_scope("LB"):
		    theta_dd_LB = LB[0].tf_eval_bound(theta_t_LB) + LB[1].tf_eval_bound(theta_t_LB) + c2_LB
		with tf.name_scope("UB"):
		    theta_dd_UB = UB[0].tf_eval_bound(theta_t_UB) + UB[1].tf_eval_bound(theta_t_UB) + c2_UB

		# Euler integration
		deltat = tf.constant([0.05], name="delta_t")

		# outputs: theta at t+1 and theta_dot at t+1
		with tf.name_scope("time_"+str(num+1)):
			theta_tp1_UB = theta_t_UB + theta_d_t_UB*deltat
			theta_tp1_LB = theta_t_LB + theta_d_t_LB*deltat
			theta_d_tp1_UB = theta_d_t_UB + theta_dd_UB*deltat
			theta_d_tp1_LB = theta_d_t_LB + theta_dd_LB*deltat
	#
	print("built dynamics graph")
	return ([theta_tp1_UB, theta_tp1_LB, theta_d_tp1_UB, theta_d_tp1_LB])

sess = tf.Session()
print("initialized session")
#sess.run(tf.global_variables_initializer())
#print("initialized global variables in dynamics")
# test_dynamics = True
# if test_dynamics:
# 	sess.run(tf.global_variables_initializer())
# 	print("theta_tp1_UB", sess.run([theta_tp1_UB]))
# 	print("theta_tp1_LB", sess.run([theta_tp1_LB]))
# 	print("theta_d_tp1_UB", sess.run([theta_d_tp1_UB]))
# 	print("theta_d_tp1_LB", sess.run([theta_d_tp1_LB]))

###########################################################
# load controller! :)
###########################################################
policy_file = "/Users/Chelsea/" # mac
policy_file = policy_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/ITWORKSrelutanh_small_network_ppo/params.pkl"

# load policy object:
with sess.as_default():
	#with tf.name_scope("Controller"):
	data = joblib.load(policy_file)
	policy = data["policy"]
	print("loaded controller")

# Controller -> Dynamics
# input of controller is theta and theta-dot
# output of controller is action....

action_UB = policy.dist_info_sym(tf.stack([theta_t_UB, theta_d_t_UB], axis=1), [])["mean"]
print("action_UB: ", sess.run([action_UB]))
#action_LB = policy.get_action([theta_t_LB, theta_d_t_LB])
#torque_UB = action_UB # < okay doing this has no bearing at all on the theta that's computed by the dynamics :/, so they're not really linked...
#torque_LB = action_LB

# input of dynamics is torque(action), output is theta theta-dot at the next timestep
print("theta_tp1_UB: ", sess.run([theta_tp1_UB]))
#print("theta_tp1_LB", sess.run([theta_tp1_LB]))
print("theta_d_tp1_UB: ", sess.run([theta_d_tp1_UB]))
#print("theta_d_tp1_LB", sess.run([theta_d_tp1_LB]))

# okay, I want to "connect" the graphs and then export to tensorbooard the graph file
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/combo_take_3"
train_writer = tf.summary.FileWriter(LOGDIR) #, sess.graph)
train_writer.add_graph(sess.graph)
train_writer.close()

# try loading just one combo (one dynamics and one controller) and looking at it in tensorboard

# (it would be cool to run it, but i also don't have to if it turns out to be too hard)









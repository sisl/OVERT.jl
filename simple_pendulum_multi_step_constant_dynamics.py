# simple pendulum problem

# theta_dd = F/ml  - gsin(theta)/l
# theta_dd = Tau/(ml^2) - (g/l)*sin(theta)
import colored_traceback.always
import tensorflow as tf 
import numpy as np
from simple_overapprox_simple_pendulum import line, bound, build_sin_approx
import os
import joblib
from rllab.sampler.utils import rollout
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import sandbox.rocky.tf.core.layers as L
from tensorflow.python.framework import graph_util

# inputs: state and action yields next state
def create_dynamics_block(num, var_dict):
	with tf.name_scope("Dynamics_"+str(num)):
		torque_UB = var_dict["action_UB"]
		state_UB = var_dict["state_UB"]
		torque_LB = var_dict["action_LB"]
		state_LB = var_dict["state_LB"]
		
		m = 0.25 # kg
		l = 0.1 # m
		oomls = tf.constant([[(1/ (m*(l**2)) )]], name="torque_scaling")

		# constant accel bounds
		# constant from dynamics plus scaled torque
		const_UB = tf.constant([[50.]])
		const_LB = tf.constant([[-50.]])

		# Euler integration
		deltat = tf.constant([[0.05]], name="delta_t")
		# outputs: state (theta and thetadot) at t+1
		with tf.name_scope("time_"+str(num+1)):
			theta_d_UB = tf.constant([[0.,1.],[0.,0.]])@state_UB
			theta_dd_UB = tf.constant([[0.],[1.]])@(oomls@torque_UB + const_UB)
			change_UB = theta_d_UB@deltat + theta_dd_UB@deltat
			state_tp1_UB = state_UB + change_UB
			# identical for lower bound
			theta_d_LB = tf.constant([[0.,1.],[0.,0.]])@state_LB
			theta_dd_LB = tf.constant([[0.],[1.]])@(oomls@torque_LB + const_LB)
			change_LB = theta_d_LB@deltat + theta_dd_LB@deltat
			state_tp1_LB = state_LB + change_LB
			# identical for lower bound

	#
	print("built dynamics graph")
	return [state_tp1_UB, state_tp1_LB]

sess = tf.Session()
print("initialized session")
# Initialize theta and theta-dot
with tf.variable_scope("initial_values"):
	state_UB = tf.Variable([[1.0],[0.0]], name="state_UB")
	# theta, theta dot, UB
	state_LB = tf.Variable([[1.0], [0.0]], name="state_LB")
sess.run(tf.global_variables_initializer())

###########################################################
# load controller! :)
###########################################################
policy_file = "/Users/Chelsea/" # mac
policy_file = policy_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/relu_small_network_vpg_capped_action_trying_simpler_dense_layer/params.pkl"

# load policy object:
with sess.as_default():
	#with tf.name_scope("Controller"):
	data = joblib.load(policy_file)
	policy = data["policy"]
	print("loaded controller")
	g = sess.graph.as_graph_def()
	[print(n.name) for n in g.node]
	output_node_name = "policy/mean_network/output"
	output_graph_def = graph_util.convert_variables_to_constants(
	    sess, # sess used to retrieve weights
	    g, # graph def used to retrieve nodes
	    output_node_name.split(",") # output node names used to select useful nodes
	    )
import pdb; pdb.set_trace()


# Controller -> Dynamics

# input of controller is theta and theta-dot
# output of controller is action....
for i in range(5):
	with tf.name_scope("get_actions"):
		#action_UB = policy.dist_info_sym(tf.transpose(state_UB), [])["mean"]
		action_UB, = L.get_output([policy._l_mean], tf.transpose(state_UB))
		print("action_UB: ", sess.run([action_UB]))
		#
		#action_LB = policy.dist_info_sym(tf.transpose(state_LB), [])["mean"]
		action_LB, = L.get_output([policy._l_mean], tf.transpose(state_LB))
		print("action_LB: ", sess.run([action_LB]))

	# input of dynamics is torque(action), output is theta theta-dot at the next timestep
	with tf.name_scope("run_dynamics"):
		var_dict = {"action_UB": action_UB, 
					"state_UB": state_UB, 
					"action_LB": action_LB, 
					"state_LB": state_LB, 
					}
		[state_UB, state_LB] = create_dynamics_block(num=i, var_dict=var_dict)
		print("[theta, theta_dot]_UB: ", sess.run([state_UB]))
		print("[theta, theta_dot]_LB", sess.run([state_LB]))
	
# okay, I want to "connect" the graphs and then export to tensorbooard the graph file
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/constant_dynamics_relu_policy_debug3"
train_writer = tf.summary.FileWriter(LOGDIR) #, sess.graph)
train_writer.add_graph(sess.graph)
train_writer.close()

# next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/UGH_multi_2


# see how many unique ops in the graph
# get current graph
# TODO: how to print all ops in a graph? below code only works for serialized graphs
g = sess.graph.as_graph_def()
# print n for all n in graph_def.node
[print(n) for n in g.node]
# make a set of the ops:
op_set = {(x.op,) for x in g.node}
# print this set of ops
[print(o[0]) for o in op_set]


# different layers of abstraction to get action at:
# import sandbox.rocky.tf.core.layers as L
# action = L.get_output([policy._l_mean], state) # med-high level
# If this doesn't do it, I may have to train a new rllab policy that uses like a tf network as the base MLP instead of the rllab MLP type...shouldn't be TOO hard? I did this kind of thing once before already?







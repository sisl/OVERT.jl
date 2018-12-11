# simple pendulum problem

# theta_dd = F/ml  - gsin(theta)/l
# theta_dd = Tau/(ml^2) - (g/l)*sin(theta)

import tensorflow as tf 
import numpy as np
from simple_overapprox_simple_pendulum import line, bound, build_sin_approx

# inputs: state and action yields next state
with tf.variable_scope("time_t"):
	torque = tf.Variable([0.0], name="Torque")
	theta_t_UB = tf.Variable([-1.0], name="theta_t_UB")
	theta_t_LB = tf.Variable([1.0], name="theta_t_LB")
	theta_d_t_UB = tf.Variable([0.0], name="theta_d_t_UB")
	theta_d_t_LB = tf.Variable([0.0], name="theta_d_t_LB")

m = 0.25 # kg
l = 0.1 # m
g = 9.8 # m/s^2
oomls = tf.constant([(1/ (m*(l**2)) )], name="torque_scaling")

c1 = g/l
c2 = torque*oomls

fun = lambda x: c1*np.sin(x)

# build_sin_approx(fun, c1, convex_reg, concave_reg)
LB, UB = build_sin_approx(fun, c1, [-np.pi, 0], [0, np.pi])

with tf.name_scope("LB"):
    theta_dd_LB = LB[0].tf_eval_bound(theta_t_LB) + LB[1].tf_eval_bound(theta_t_LB) + c2
with tf.name_scope("UB"):
    theta_dd_UB = UB[0].tf_eval_bound(theta_t_UB) + UB[1].tf_eval_bound(theta_t_UB) + c2

# Euler integration
deltat = tf.constant([0.05], name="delta_t")

# outputs: theta at t+1 and theta_dot at t+1
with tf.name_scope("time_tp1"):
	theta_tp1_UB = theta_t_UB + theta_d_t_UB*deltat
	theta_tp1_LB = theta_t_LB + theta_d_t_LB*deltat
	theta_d_tp1_UB = theta_d_t_UB + theta_dd_UB*deltat
	theta_d_tp1_LB = theta_d_t_LB + theta_dd_LB*deltat

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([theta_tp1_UB, theta_tp1_LB, theta_d_tp1_UB, theta_d_tp1_LB]))

LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/td1"
train_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
train_writer.add_graph(sess.graph)
train_writer.close()


# Thing to debug: Why when I load the graph into tensorboard does it say "found more than one graph event per run" --> do I need to pass the scope or session to the functions in build_sin_approx in order to keep everything in one graph?









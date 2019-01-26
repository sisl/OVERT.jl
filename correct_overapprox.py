# adapt simple_pendulum_multi_step_constant_dynamics.py
# make it simpler
# make it a true overapproximation, so that output set is large enough to be correct

# Features: __________________________
# multiple steps
# constant dynamics (accel) bound


import colored_traceback.always
import tensorflow as tf 
import numpy as np
import os
import joblib

# parsing support libs
from tensorflow.python.framework import graph_util
import parsing
from NNet.scripts.writeNNet import writeNNet

verbose = True
run = True
f_id = str(int(np.round(np.random.rand()*5000)))

# implement new dynamics block

# Q: once parsed, how to keep track of which inputs are which? need to name them maybe, and then name them when parsing too? or at the very least somehow collect the order of the output names when parsing, and the order of the input names 

# plan: treat theta and thetadot seperately until i need to feed them into controller, where you can multiply by tall matrices and add together to concat

class Dynamics():
    def __init__(self): 
        with tf.name_scope("dynamics_constants"):
            self.m = 0.25 # kg
            self.l = 0.1 # m
            self.oomls = tf.constant([[(1/ (self.m*(self.l**2)) )]], name="torque_scaling")
            # currently accel bounds are constant but will be funcs of theta
            self.accel_UB = lambda theta: tf.constant([[50.]])
            self.accel_LB = lambda theta: tf.constant([[-50.]])
            self.deltat = tf.constant(0.05*np.eye(1), dtype='float32', name="deltat")

    def run(self, num, action, theta, theta_dot): 
        with tf.name_scope("Dynamics_"+str(num)):
            theta_hat = tf.add(theta, self.deltat@theta_dot, name="theta_"+str(num))
            theta_dot_UB = tf.add(
                            theta_dot,
                            self.deltat@(self.oomls@action + self.accel_UB(theta)),
                            name="theta_dot_UB_"+str(num)
                        )
            theta_dot_LB = tf.add(
                            theta_dot,
                            self.deltat@(self.oomls@action + self.accel_LB(theta)),
                            name="theta_dot_LB_"+str(num)
                        )
        return theta_hat, theta_dot_LB, theta_dot_UB

class Controller():

    def __init__(self):
        with tf.name_scope('Controller'):
            self.W0 = tf.Variable(np.random.rand(4,2)*2-1, dtype='float32')
            self.b0 = tf.Variable(np.random.rand(4,1)*2-1, dtype='float32')
            self.W1 = tf.constant(np.random.rand(4,4)*2-1, dtype='float32')
            self.b1 = tf.constant(np.random.rand(4,1)*2-1, dtype='float32')
            self.Wf = tf.constant(np.random.rand(1,4)*2-1, dtype='float32')
            self.bf = tf.constant(np.random.rand(1,1)*2-1, dtype='float32')
    def run(self, state0):
        state = tf.nn.relu(self.W0@state0 + self.b0)
        state = tf.nn.relu(self.W1@state + self.b1)
        statef = self.Wf@state + self.bf
        return statef

class ReluProtector():
    def __init__(self, state_dim):
        self.before = tf.constant(np.vstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
        self.after = tf.constant(np.hstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
    def apply(self, state, name=""):
        return tf.matmul(self.after, tf.nn.relu(self.before@state), name=name)

def build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, ncontroller_act):
    with tf.name_scope("assign_init_vals"):
        relu_protector = ReluProtector(1)
        theta = theta_0
        theta_dot = theta_dot_0
        theta_dot_hats = []
        for i in range(nsteps):
            theta_dot_hats.append(tf.placeholder(
                    tf.float32,
                    shape=(1,1),
                    name="theta_dot_hat_"+str(i+1)
                ))
        theta_dot_LBs = []
        theta_dot_UBs = []
    #
    for i in range(nsteps):
        with tf.name_scope("get_action"):
            state = tf.constant([[1.],[0.]])@theta + tf.constant([[0.],[1.]])@theta_dot 
            action = controller.run(state)
        #
        # apply relu protection
        with tf.name_scope("relu_protection"):
            for j in range(ncontroller_act):
                theta = relu_protector.apply(theta, name="theta")
                theta_dot = relu_protector.apply(theta_dot, name="theta_dot")
                theta_dot_LBs = [relu_protector.apply(tdlb, name="tdlb") for tdlb in theta_dot_LBs]
                theta_dot_UBs = [relu_protector.apply(tdub, name="tdub") for tdub in theta_dot_UBs]
                theta_dot_hats = [relu_protector.apply(tdh, name="tdh") for tdh in theta_dot_hats]
            
        # input of dynamics is torque(action), output is theta theta-dot at the next timestep
        with tf.name_scope("run_dynamics"):
            theta, theta_dot_LB, theta_dot_UB = dynamics.run(num=i+1, action=action, theta=theta, theta_dot=theta_dot)
            theta_dot_LBs.append(theta_dot_LB)
            theta_dot_UBs.append(theta_dot_UB)
            theta_dot = theta_dot_hats[i]
            
    #
    return (theta, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs)

def display_ops(sess):
    g = sess.graph.as_graph_def()
    if verbose:
            # print n for all n in graph_def.node
            print("before freezing:")
            [print(n.name) for n in g.node]
            # make a set of the ops:
            op_set = {(x.op,) for x in g.node}
            # print this set of ops
            [print(o[0]) for o in op_set]

def write_to_tensorboard(logdir, sess):
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(sess.graph) 
    train_writer.close()
    print("wrote to tensorboard log")

# build model
with tf.variable_scope("initial_values"):
    theta_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_0")
    theta_dot_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_dot_0")
controller = Controller()
ncontroller_act = 2
nsteps = 3
dynamics = Dynamics()
theta, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs = build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, ncontroller_act)
with tf.name_scope("outputs"):
    theta_out = tf.constant([[0.0]])@theta_dot + theta

# run stuff
sess = tf.Session()
print("initialized session")
init_theta = np.array([[1.0]])
init_theta_dot = np.array([[-1.0]])
display_ops(sess)
feed_dict = {
        theta_0: init_theta,
        theta_dot_0: init_theta_dot,
        'assign_init_vals/theta_dot_hat_1:0':np.array([[0.1]]),
        'assign_init_vals/theta_dot_hat_2:0':np.array([[0.2]]),
        'assign_init_vals/theta_dot_hat_3:0':np.array([[0.3]]),
    }
sess.run(tf.global_variables_initializer())
theta_v, theta_dot_v, theta_dot_hats_v, theta_dot_LBs_v, theta_dot_UBs_v = sess.run([theta, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs], feed_dict=feed_dict)
print("theta: ", theta_v)
print("theta_dot: ", theta_dot_v)
print("theta_dot_hats: ", theta_dot_hats_v)
print("theta_dot_LBs: ", theta_dot_LBs_v)
print("theta_dot_UBs: ", theta_dot_UBs_v)

# print output op names
print("theta, theta_dot")
print([t.op.name for t in [theta, theta_dot]])
print("theta_dot_hats")
print([t.op.name for t in theta_dot_hats])
print("theta_dot_LBs")
print([t.op.name for t in theta_dot_LBs])
print("theta_dot_UBs")
print([t.op.name for t in theta_dot_UBs])


# write to tensorboard
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/new_approach_"+f_id
write_to_tensorboard(LOGDIR, sess)
# next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/UGH_multi_2

# parse and eval and turn into .nnet and set up inequalities in NV.jl
# first convert variables to constants
output_node_names = ['run_dynamics_2/Dynamics_3/theta_3', # theta
                    'relu_protection_2/tdh_3', 
                    'relu_protection_2/tdh_4', 
                    'relu_protection_2/tdh_5', # theta dots
                    'relu_protection_2/tdlb_2', 
                    'relu_protection_2/tdlb_3', 
                    'run_dynamics_2/Dynamics_3/theta_dot_LB_3', # theta dot LBs
                    'relu_protection_2/tdub_2', 
                    'relu_protection_2/tdub_3', 
                    'run_dynamics_2/Dynamics_3/theta_dot_UB_3', # theta dot UBs
                    ]
output_graph_def = graph_util.convert_variables_to_constants(
        sess, # sess used to retrieve weights
        sess.graph.as_graph_def(), # graph def used to retrieve nodes
        output_node_names # output node names used to select useful nodes
        )
# print op list to make sure its only stuff we can handle
print("op set: ", {(x.op,) for x in output_graph_def.node})

# turn graph def back into graph
tf.import_graph_def(output_graph_def)
display_ops(sess)
g = tf.get_default_graph()

# get output ops
output_ops = [g.get_operation_by_name("import/"+node) for node in output_node_names]

activation_type = 'Relu'
W,b = parsing.parse_network_wrapper(output_ops, activation_type, sess)
# order of input ops: 
# ['import/initial_values/theta_dot_0', 'import/assign_init_vals/theta_dot_hat_1', 'import/initial_values/theta_0', 'import/assign_init_vals/theta_dot_hat_2', 'import/assign_init_vals/theta_dot_hat_3']









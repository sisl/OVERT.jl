import colored_traceback.always
from correct_overapprox import Dynamics, Controller, ReluProtector, build_multi_step_network, display_ops, write_to_tensorboard, write_metadata, collect_output_ops
import tensorflow as tf 
import numpy as np
import os
import joblib

#######################################
# 3 steps
#######################################

# parsing support libs
from tensorflow.python.framework import graph_util
import parsing
from NNet.scripts.writeNNet import writeNNet

verbose = True
run = True
f_id = str(int(np.round(np.random.rand()*5000)))

# build model
with tf.variable_scope("initial_values"):
    theta_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_0")
    theta_dot_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_dot_0")
controller = Controller()
ncontroller_act = 2 # number of controller activations along any one path through controller per... time applied?
nsteps = 3
dynamics = Dynamics()
## 
thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs = build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, ncontroller_act)
with tf.name_scope("outputs"):
    theta_out = tf.constant([[0.0]])@theta_dot + thetas[-1]

# run stuff
sess = tf.Session()
print("initialized session")
init_theta = np.array([[1.0]])
init_theta_dot = np.array([[-1.0]])
init_theta_dot_hat_1 = np.array([[0.1]])
init_theta_dot_hat_2 = np.array([[-0.2]])
init_theta_dot_hat_3 = np.array([[0.3]])
display_ops(sess)
feed_dict = {
        theta_0: init_theta,
        theta_dot_0: init_theta_dot,
        'assign_init_vals/theta_dot_hat_1:0': init_theta_dot_hat_1,
        'assign_init_vals/theta_dot_hat_2:0': init_theta_dot_hat_2,
        'assign_init_vals/theta_dot_hat_3:0': init_theta_dot_hat_3,
    }
sess.run(tf.global_variables_initializer())
thetas_v, theta_dot_v, theta_dot_hats_v, theta_dot_LBs_v, theta_dot_UBs_v = sess.run([thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs], feed_dict=feed_dict)
print("thetas: ", thetas_v)
print("theta_dot: ", theta_dot_v)
print("theta_dot_hats: ", theta_dot_hats_v)
print("theta_dot_LBs: ", theta_dot_LBs_v)
print("theta_dot_UBs: ", theta_dot_UBs_v)

import pdb; pdb.set_trace()
output_ops = collect_output_ops([thetas, theta_dot_hats, theta_dot_LBs, theta_dot_UBs])
output_op_names = [op.name for op in output_ops]
print("output_ops: ", output_op_names)

# parse and eval and turn into .nnet and set up inequalities in NV.jl
# first convert variables to constants
output_graph_def = graph_util.convert_variables_to_constants(
        sess, # sess used to retrieve weights
        sess.graph.as_graph_def(), # graph def used to retrieve nodes
        output_op_names # output node names used to select useful nodes
        )
# print op list to make sure its only stuff we can handle
print("op set: ", {(x.op,) for x in output_graph_def.node})
# turn graph def back into graph
tf.import_graph_def(output_graph_def)
#display_ops(sess)
g = tf.get_default_graph()

# get output ops
output_ops = [g.get_operation_by_name("import/"+node) for node in output_op_names]

activation_type = 'Relu'
W,b, input_list = parsing.parse_network_wrapper(output_ops, activation_type, sess)
# order of input ops: 
input_dict = {'import/initial_values/theta_dot_0':init_theta_dot, 
                'import/assign_init_vals/theta_dot_hat_1':init_theta_dot_hat_1, 
                'import/initial_values/theta_0':init_theta, 
                'import/assign_init_vals/theta_dot_hat_2':init_theta_dot_hat_2, 
                'import/assign_init_vals/theta_dot_hat_3':init_theta_dot_hat_3}

# turn into its own tf network to check equivalency
W.reverse()
b.reverse()
ff_input = tf.placeholder(tf.float32, shape=(5,1), name="state0")
ffnet = parsing.create_tf_network(W,b,inputs=ff_input, activation=tf.nn.relu, act_type='Relu', output_activated=False)

# write to tensorboard
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/new_approach_"+f_id
write_to_tensorboard(LOGDIR, sess)
# next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/new_approach_3605


feed_dict = {
                ff_input: np.vstack([input_dict[i] for i in input_list])
            }

output_tensors_v = sess.run(ffnet, feed_dict=feed_dict)
print("condensed output: ", output_tensors_v)
assert(all(abs(np.array(thetas_v).flatten() - output_tensors_v[0:3].flatten())<1e-4)) # theta
assert(all(abs(np.array(theta_dot_hats_v).flatten() - output_tensors_v[3:6].flatten())<1e-4)) # theta dot hats
assert(all(abs(np.array(theta_dot_LBs_v).flatten() - output_tensors_v[6:9].flatten())<1e-4)) # theta dot LB
assert(all(abs(np.array(theta_dot_UBs_v).flatten() - output_tensors_v[9:12].flatten())<1e-4))# theta dot UBs

# write to .nnet file
# and write accompanying file with meta data like order of inputs and outputs
means = [0.,0.,0.,0.,0.,0.]
ranges = [1., 1., 1., 1., 1., 1.]
inMins = [-1.,-50., -1., -50., -100.]
inMaxs = [1.,50., 1., 50., 100.]
directory = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files/"
fileName = os.path.join(directory, "correct_overrapprox_const_dyn_"+f_id+".nnet")
writeNNet(W,b,inputMins=inMins,inputMaxes=inMaxs,means=means,ranges=ranges, order='Wx', fileName=fileName)

ff_oo_name = ffnet.op.name
output_tensors = [i.name for o in output_ops for i in o.outputs]
write_metadata(input_list, output_tensors, ff_oo_name, directory, f_id)

# write ff tf network to file
ff_output_graph_def = graph_util.convert_variables_to_constants(
        sess, # sess used to retrieve weights
        sess.graph.as_graph_def(), # graph def used to retrieve nodes
        [ffnet.op.name] # output node names used to select useful nodes
        )
output_graph_name = "nnet_files/graph_def_"+f_id+".pb"
with tf.gfile.GFile(output_graph_name, "w") as f:
    f.write(ff_output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(ff_output_graph_def.node))

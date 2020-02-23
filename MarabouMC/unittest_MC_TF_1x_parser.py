################################################################
## This is for tf 1.x
## Feb. 17 2020
## Chelsea Sidrane
################################


import tensorflow as tf # 1.x
import numpy as np
from write_graphdef import write_graphdef
from MC_TF_parser import TFConstraint
import os
import colored_traceback.always

# create network
sess = tf.Session()
with sess.as_default():
    x = tf.placeholder(shape=(2,1), dtype='float64')
    W = np.random.rand(1,2)
    b = np.random.rand(1,1)
    output = tf.nn.relu(tf.matmul(W,x) + b)

# write graph
filename = write_graphdef(sess, output)

# read graphdef
tfconstraints = TFConstraint(filename, inputNames=[x.op.name], outputName=output.op.name)

# test that output of tf network satisfies constraints of TFConstraint
# pick random inputs, run with tf graph, get outputs
inputx = np.random.rand(2,1)
# eval with constraints
cfeed = dict(zip(tfconstraints.inputVars[0].flatten().tolist(), inputx.flatten().tolist()))
solutions = tfconstraints.eval_constraints(cfeed)
#print(solutions)
output_sols = np.array([solutions[ov[0]] for ov in tfconstraints.outputVars], dtype='float64')
print(output_sols)

# eval with tf
tfsol = sess.run(output, feed_dict={x: inputx}).flatten()
print(tfsol)

# assert that they are the same!
assert(all(np.isclose(output_sols, tfsol)))


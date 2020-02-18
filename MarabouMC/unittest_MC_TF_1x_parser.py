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
    x = tf.placeholder(shape=(5,1), dtype='float64')
    W = np.random.rand(3,5)
    b = np.random.rand(3,1)
    output = tf.nn.relu(tf.matmul(W,x) + b)

# write graph
filename = write_graphdef(sess, output)

# read graphdef
tfconstraints = TFConstraint(filename, inputNames=[x.op.name], outputName=output.op.name)
import pdb; pdb.set_trace()
print("hi")

# test that output of tf network satisfies constraints of TFConstraint
# pick random inputs, run with tf graph, get outputs
# 
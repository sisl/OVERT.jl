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

# create random network controller
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

# do MC! ######################################################################################
from dynamical_systems.dynamical_systems import f1
from MC_interface import OVERTDynamics, TFControlledOVERTTransitionRelation, MyTransitionSystem, BMC, TFController
from marabou_interface import MarabouWrapper

my_dynamics = OVERTDynamics(f1, np.array([["x"], ["theta"]], dtype=object), np.array([["u"]], dtype=object))
init_set = {"x": (0, 5), "theta": (-np.pi/4, np.pi/4)}
controller = TFController(tf_sess = sess, inputNames=[x.op.name], outputName=output.op.name)
tr = TFControlledOVERTTransitionRelation(dynamics_obj=my_dynamics, controller_obj=controller)
transition_system = MyTransitionSystem(states=my_dynamics.states, initial_set=init_set, transition_relation=tr)

solver = MarabouWrapper()

property_file = "my_prop1.txt" # property defined over the states
algo = BMC(ts = transition_system, prop_file = property_file, solver=solver)
import pdb; pdb.set_trace()
algo.check_invariant_until(3)

# for debug
# imp.reload(MC_interface) 
# from MC_interface import OVERTDynamics, TFControlledOVERTTransitionRelation, MyTransitionSystem, BMC, TFController

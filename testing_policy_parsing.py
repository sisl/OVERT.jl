import tensorflow as tf
import numpy as np
import joblib
from rllab.sampler.utils import rollout
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from correct_overapprox import ReluProtector, build_multi_step_network, display_ops, write_to_tensorboard, write_metadata, collect_output_ops
from tensorflow.python.framework import graph_util
from NNet.scripts.pb2nnet import FFTF2W, pb2W
import parsing


# Goal of script: parse 'xW' convention rllab (feedforward) policies into 'Wx' networks that I can then combine with dynamics easily

# file = "/Users/Chelsea/Dropbox/AAHAA/src/rllab/data/local/experiment/relu_small_network_vpg_capped_action_trying_simpler_dense_layer/params.pkl"
# # ^ the only policy file that works. ALSO...I think I messed up my rllab, trrying to change the weights.....so I have a bit to fix before I can make new policies again

file = "/Users/Chelsea/Dropbox/AAHAA/src/rllab/data/local/experiment/relu_small_network_ppo_capped_action/params.pkl"

sim = False

# load and simulate
sess = tf.Session()
# load policy
with sess.as_default():
    data = joblib.load(file)
    policy = data["policy"]
    # can sim policy to test it! :DDD
    if sim:
        env = TfEnv(GymEnv("MyPendulum-v0", record_video=False))
        path = rollout(env, policy, max_path_length=500, animated=True, speedup=2, always_return_paths=True)
        print("reward: ", sum(path["rewards"]))
        input("enter to continue")


# extract mean network only, make sure it looks ff
output_op_names = ["policy/mean_network/output"]
output_graph_def = graph_util.convert_variables_to_constants(
        sess, # sess used to retrieve weights
        sess.graph.as_graph_def(), # graph def used to retrieve nodes
        output_op_names # output node names used to select useful nodes
        )
# print op list to make sure its only stuff we can handle
print("op set: ", {(x.op,) for x in output_graph_def.node})
# turn graph def back into graph
tf.import_graph_def(output_graph_def)


# write to tensorboard
f_id = str(int(np.round(np.random.rand()*5000)))
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/looking_at_old_controller_"+f_id
write_to_tensorboard(LOGDIR, sess)
# next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/looking_at_old_controller_4909


# save as .pb
output_graph_name = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/nnet_files/bad_relu_rllab_controller_graph_def_"+f_id+".pb"
with tf.gfile.GFile(output_graph_name, "w") as f:
    f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node)) 

# get out weights and biases for turning back into a tf network
weights, biases = pb2W(output_graph_name, inputName="policy/mean_network/input/input", outputName=output_op_names[0])
# have to transpose because we're coming from rllab
weights  = [w.transpose() for w in weights]
# NOTE: only doing this next step like this because I am using a network that has 'expand dims' in it, and thus I pick out the 1D biases from rllab
biases = [np.array([b]).transpose() for b in biases]

# turn back into a tf network
ff_input = tf.placeholder(tf.float32, shape=(2,1), name="state0")
ffnet = parsing.create_tf_network(weights,biases,inputs=ff_input, activation=tf.nn.relu, act_type='Relu', output_activated=False)

# evaluate to make sure that the networks are the same
rllab_input = tf.placeholder(tf.float32, shape=(1,2), name="rllab_inputs")
test_input_xW = np.random.rand(1,2)*2-1
test_input_Wx = test_input_xW.transpose()
rllab_output_tensor = tf.get_default_graph().get_operation_by_name(output_op_names[0]).outputs[0]
rllab_output = sess.run([rllab_output_tensor], feed_dict={"policy/mean_network/input/input:0": test_input_xW})
ffnet_output = sess.run([ffnet], feed_dict={ff_input: test_input_Wx})

assert all(rllab_output[0] - ffnet_output[0]<1e-3)
print("tests pass!") 





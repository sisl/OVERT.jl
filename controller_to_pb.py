import tensorflow as tf
import joblib
import correct_overapprox as co
import numpy as np

policy_file = "/Users/Chelsea/"
policy_file = policy_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/"
policy_name = "convnet_1_ppo/"
policy_file = policy_file + policy_name + "params.pkl"

sess = tf.Session()
with sess.as_default():
    data = joblib.load(policy_file)
    #self.policy = data["policy"]
    print("loaded controller")


tensorboard_log_dir = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/" + policy_name

#co.write_to_tensorboard(tensorboard_log_dir, sess)
# tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/convnet_1_ppo --host=localhost --port=1234

output_op_name = "policy/nonlinearity_2/output"
input_op_name = "policy/mean_network/input/input"

f_id = str(1)
network_dir = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/vision_related"
# export as frozen model
co.write_pb_core(sess, [output_op_name], network_dir, f_id, verbose=True)


# export as metagraph
tf.train.export_meta_graph(
	network_dir + "/meta_graph.pb",
	collection_list = [output_op_name, input_op_name]
	)
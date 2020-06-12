# excellent policy to nnv
import joblib
import tensorflow as tf
import numpy as np

directory = "../rllab/data/local/experiment/EXCELLENT_POLICY_relu_small_network_ppo_capped_action_simpler_dense_layer_xW_learn_std_smaller_learning_rate/"
file = directory + "params.pkl"

with tf.Session() as sess:
    data = joblib.load(file)
    policy = data['policy']
    W = {}
    b = {}
    info = {}
    elems = 0
    layer_size = []
    for i in range(3):
        L = "policy/mean_network/DenseLayer2_" + str(i)
        Weight = sess.graph.get_operation_by_name(L + "/W")
        Bias = sess.graph.get_operation_by_name(L + "/b")
        W[i] = Weight.outputs[0].eval(session=sess).transpose()
        b[i] = Bias.outputs[0].eval(session=sess).transpose()
        elems = elems + np.prod(W[i].shape) + np.prod(b[i].shape)
        layer_size.append(W[i].shape[0])
    info['number_of_weights'] = elems
    info['number_of_outputs'] = b[2].shape[0]
    info['number_of_inputs'] = W[0].shape[1]
    info['number_of_layers'] = 3
    info['layer_size'] = layer_size
    info['act_functions'] = ['relu', 'relu', 'linear']
    info['max'] = ['deprecated', 'deprecated']
    info['min'] = ['deprecated', 'deprecated']

f = open(directory + "nnv.txt", 'w')
for i in range(len(W)):
    f.write("W"+str(i)+"\n")
    for row in range(W[i].shape[0]):
        for col in range(W[i].shape[1]):
            f.write(str(W[i][row, col])+", ")
        f.write("\n")

for i in range(len(b)):
    f.write("b"+str(i)+"\n")
    for row in range(b[i].shape[0]):
        f.write(str(b[i][row, 0]) + ", ")
        f.write("\n")

f.write("Info: \n")
for k in info.keys():
    f.write(k + "\n")
    f.write(str(info[k]) + "\n")

f.close()


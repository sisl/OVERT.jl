import tensorflow as tf
import numpy as np

def graph1():
	x = tf.Variable([[0.0], [0.0]])
	y = tf.Variable([[1.0], [1.0]])
	W1 = tf.constant([[2.0, 0.0],[0.0, 2.0]])
	b1 = tf.constant([[-1.0], [-1.0]])
	W2 = -W1
	b2 = -b1
	with tf.name_scope("matmul_1"):
		z1 = W1@x + b1
	with tf.name_scope("matmul_2"):
		z2 = W2@y + b2
	return z1,z2

R = tf.Variable([[1.0],[1.0]])
C = tf.constant([[-1.0], [-1.0]])
W1 = tf.constant(2*np.eye(2), dtype='float32')
b1 = tf.constant([[5.0], [5.0]])
W2 = tf.constant(3*np.eye(2), dtype='float32')
b2 = tf.constant([[7.0], [7.0]])

Q = R + C
x= W1@Q + b1
y = W2@R + b2
F = x + y

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def write(sess):
	LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/simple_parsing"
	train_writer = tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)
	train_writer.close()

import parsing as p

p.parse_network([F.op], [], [], [], [], tf.nn.relu, sess)








import tensorflow as tf
import numpy as np
import colored_traceback.always

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

def network_to_test_case_4(sess):
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

	sess.run(tf.global_variables_initializer())

	correct_mats = []
	correct_mats.append(np.hstack([np.eye(2), np.eye(2)]))
	correct_mats.append(np.vstack([
			np.hstack([np.eye(2), np.zeros((2,2))
				]),
			np.hstack([np.zeros((2,2)), np.eye(2)
				])
		]))
	with sess.as_default():
		correct_mats.append(np.vstack([
							np.hstack([
								W1.eval(),
								np.zeros((2,2))
								]), 
							np.hstack([
								np.zeros((2,2)),
								W2.eval()
								])
							])
						)
	correct_mats.append(np.vstack([
							np.hstack([np.eye(2), np.zeros((2,2)) ]),
							np.hstack([np.zeros((2,2)), np.eye(2)])
								  ])
					   )
	correct_mats.append(np.vstack(
							[np.eye(2),np.eye(2)]
		)
	)
	with sess.as_default():
		correct_biases = [
			np.zeros((2,1)),
			# the add
			np.vstack([
				b1.eval(),
				b2.eval()
				]),
			# the multiply
			np.zeros((4,1))
			# the constant add
			np.vstack([
				C.eval(),
				np.zeros((2,1))
				]),
			np.zeros((4,1))

		]
	return F, correct_mats, correct_biases

def network_to_test_case_3(sess):
	F, M, B = network_to_test_case_3(sess)
	
	

def network_to_handle_duplicates_immediately_before_relus():
	pass

# AKA duplication preceding a mix of variables, activations, and matmuls
def network_to_handle_duplicates_prior_to_mix():
	pass

########################################################
sess = tf.Session()

# testing zone
Output, M, B = network_to_handle_duplicates_as_variables(sess)

def write(sess):
	LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/simple_parsing"
	train_writer = tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)
	train_writer.close()

import parsing as p

p.parse_network([Output.op], [], [], [], [], tf.nn.relu, sess)








import tensorflow as tf

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/simple_parsing"
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.close()
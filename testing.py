# more testing code
import numpy as np
import tensorflow as tf
import parsing

# test: condense_list
def condense_list_test():
	W = []
	b = []
	for i in range(2):
		W.append(np.random.rand(2,2))
		b.append(np.random.rand(2,1))

	x = np.array([[1.0],[1.0]])

	# hand multiply:
	y = x
	for i in range(2):
		y = W[i]@y + b[i]

	W.reverse()
	b.reverse()
	Wc, bc = parsing.condense_list(W,b)
	yc = Wc@x + bc

	print("yc: ", yc)
	print("y: ", y)
	assert( all((yc-y) < 1e-6) )

# test: parsing!!! ########################################
# create network
def two_layer(p):
	W = tf.Variable(2*np.eye(2), dtype='float32')
	x1 = tf.nn.relu(W@p)
	W2 = tf.constant(np.eye(2), dtype='float32')
	q = tf.nn.relu(W2@x1)
	return [q]

def split(p):	
	A = tf.nn.relu(p)
	B = tf.nn.relu(p)
	return [A,B]

def mul_split(p):
	W = tf.constant(np.eye(2), dtype='float32')
	y = W@p
	A = tf.nn.relu(y)
	B = tf.nn.relu(y)
	return [A,B]

def concat(p):
	W = tf.constant(np.eye(2), dtype='float32')
	x = W@p
	y = W@p
	F = x+y
	return [F]

def skip(p):
	C = tf.constant([[1.0],[1.0]])
	Q = p+C
	W = tf.constant(np.eye(2), dtype='float32')
	x = W@Q
	F = Q + x 
	return [F]

def more_complex1(p):
	C = tf.constant([[1.0],[1.0]])
	Q = p + C
	W = tf.Variable(np.eye(2), dtype='float32')
	b = tf.constant(np.zeros((2,1)), dtype='float32')
	W2 = tf.constant(np.eye(2), dtype='float32')
	b2 = tf.constant(np.zeros((2,1)), dtype='float32')
	x = W@Q + b
	y = W2@p + b2
	F = x+y
	return [F]

def more_complex2(p):
	C = tf.constant([[1.0],[1.0]])
	Q = p + C
	W = tf.Variable(np.random.rand(2,2), dtype='float32')
	b = tf.constant(np.ones((2,1)), dtype='float32')
	W2 = tf.constant(np.random.rand(2,2), dtype='float32')
	b2 = tf.constant(np.ones((2,1)), dtype='float32')
	x = W@Q + b
	y = W2@p + b2
	F = x+y
	return [F]

def more_complex3(p):
	C = tf.constant([[1.0],[1.0]])
	Q = tf.nn.relu(p + C)
	W = tf.Variable(np.random.rand(2,2), dtype='float32')
	b = tf.constant(np.ones((2,1)), dtype='float32')
	W2 = tf.constant(np.random.rand(2,2), dtype='float32')
	b2 = tf.constant(np.ones((2,1)), dtype='float32')
	x = tf.nn.relu(W@Q + b)
	y = tf.nn.relu(W2@tf.nn.relu(p) + b2)
	F = x+y
	return [F]

p = tf.placeholder(shape=(2,1), dtype='float32')
pval = np.array([[-2.0],[2.0]])

for fn in [two_layer, split, mul_split, concat, skip, more_complex1, more_complex2, more_complex3]:

	q_list = fn(p) #skip(p) #two_layer(p)
	activation = tf.nn.relu #tf.identity
	act_type = 'Relu'
	oa = all([q.op.type=='Relu' for q in q_list])

	# eval by itself
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	q_out, = sess.run([tf.concat(q_list, axis=0)], feed_dict={p:pval})
	print("q: ", q_out)

	# squish and then eval
	W,b = parsing.parse_network([q.op for q in q_list], [], [], [], [], 'Relu', sess)
	W.reverse()
	b.reverse()
	net = parsing.create_tf_network(W,b,inputs=p, activation=activation, act_type=act_type, output_activated=oa)
	#with sess.as_default():

	net, = sess.run([net], feed_dict={p:pval})
	print("q (through parser): ", net)

	assert all( abs(q_out - net) < 1e-6)
	print("Test passes ---------------------")

print("~ ~ ~ All tests pass! ~ ~ ~")



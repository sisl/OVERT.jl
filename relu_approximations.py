# define new op

import tensorflow as tf
import numpy as np

# neg -> matmultiply by -1
# sub -> matmultiply by -1 and add

# turn max and min into relus
def ReLu(x):
    return np.maximum(x,0.0)

def ReLuMax(x,y):
    return ReLu(x-y) + y

def tfReLuMax(x,y, name=""):
    return tf.add(tf.nn.relu(x-y), y, name=name+"tfrelu_0")

def ReLuMin(x,y):
    return -ReLuMax(-x,-y)

def tfReLuMin(x,y, name=""):
    return tf.multiply(-1.0, tfReLuMax(-x,-y), name=name+"tfrelu_1")

# can be implemented as min(relu(x),constant)
# want to preserve activity between -1.0 and 1.0
def relu_tanh(x, name=""):
	return tfReLuMin(tf.nn.relu(x+1.0)-1.0, 1.0, name=name)

def relu_sigmoid(x, name=""):
	return tfReluMin(tf.nn.relu(0.25*x + 0.5),1.0, name=name)

def get_identity(x):
	if x.shape[0].value > 1:
		longdim = 0
	elif x.shape[1].value >1:
		longdim = 1
	elif x.shape[0].value == 1 and x.shape[1].value == 1:
		longdim = 1
	else:
		raise NotImplementedError
	return longdim, np.eye(x.shape[longdim].value)

def negate_tensor(x):
	"""
	Negate tensor with matmul times negative identity
	"""
	# if type(x) in [int, float]:
	# 	return -1*x
	longdim, eye = get_identity(x)
	neye = tf.constant(-1*eye, dtype='float32')
	if longdim == 0:
		negx = neye@x
	elif longdim == 1:
		negx = x@neye
	else:
		raise NotImplementedError
	return negx

def min_from_max(x, y, name=""):
	"""
	Construct a min op out of max
	"""
	# if (type(x) == tf.Tensor) and (type(y) == tf.Tensor): 
	# 	assert x.shape[0].value == y.shape[0].value
	# 	assert x.shape[1].value == y.shape[1].value
	return negate_tensor(tf.maximum(negate_tensor(x),negate_tensor(y)))

def linearized_tanh(x, name="linearized_tanh"):
	"""
	Linearized tanh in terms of max.
	"""
	one = tf.constant([[1.0]])
	neg_one = tf.constant([[-1.0]])
	with tf.name_scope(name):
		return tf.maximum(min_from_max(x, one), neg_one)





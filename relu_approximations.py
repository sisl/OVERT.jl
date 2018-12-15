# define new op

import tensorflow as tf
from OverApprox.simple_overapprox_simple_pendulum import tfReLuMax, tfReLuMin

# can be implemented as min(relu(x),constant)
# want to preserve activity between -1.0 and 1.0
def relu_tanh(x, name=""):
	return tfReLuMin(tf.nn.relu(x+1.0)-1.0, 1.0, name=name)

def relu_sigmoid(d, name=""):
	return tfReluMin(tf.nn.relu(0.25*x + 0.5),1.0, name=name)
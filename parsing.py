# parsing code
# right now, assume that we DONT take batches. we only take one "datapoint" at a time
# assume multiplication of the form: W*x + b
import numpy as np

# Do I want to create a new TF network? Or just go directly to .nnet?
# I think I want to condense to a new tf network, because then I can run some test inputs through both new and old and make sure they're the same...
# I guess I could do that once its in .nnet format too?? (evluate it)
# so there's no huge difference.
# do whatever is easiest...

# activations.....
# changliu said there's an abstract type for activation...and that I could just specify it as a function
# this means that I could specify it as a function that is "different for every layer" of my network...
# meaning all different kinds of acttivations could be supported....including idenity activation...
# hm but counting the number of times it's called...isn't a...clean...choice
# I think I should aim for having a single relu activation each layer

def are_inputs_vars(op):
	return [is_variable(i) for i in op.inputs]

# maybe keep a list of weights and biases through the recursion?
# inputs: final op
# outputs: list of weights and biases for a flattened representation of the network 
def parse_network(op, temp_W, temp_b, final_W, final_b, activation_type):
	if not (op.type==activation_type):
		# if there is more than 1 variable input
		if sum(are_inputs_vars(op)) > 1:

		else: # if there is only 1 variable input

	else # if op is of type activation
		# multiply temporary tensor list togeter to produce final tensor. add the final tensor list
		# record activation type? into activation list?
		# potentially record into .nnet file?
		# recurse on input

# find dim of vector that is not 1
# assume 2D only
def get_long_len(tfvec):
	s = tfvec._shape_as_list()
	if s[0] == 1:
		return s[1]
	else:
		return s[0]

def get_add_mat(op, sess):
	n = get_long_len(op.inputs[0])
	nprime = get_long_len(op.inputs[1])
	assert n==nprime
	if is_variable(op.inputs[0]) and is_variable(op.inputs[1]):
		W11 = np.identity(n)
		W12 = np.identity(n)
		W = np.hstack([W11, W12])
		b = np.zeros([n,1])
	elif is_variable(op.inputs[0]):
		W = np.identity(n)
		with sess.as_default():
			b = op.inputs[1].eval()
	elif is_variable(op.inputs[1]):
		W = np.identity(n)
		with sess.as_default():
			b = op.inputs[0].eval()
	else: # neither is a variable. don't expect to use this case...
		print("Why are we here?")
		W11 = np.identity(n)
		W12 = np.identity(n)
		W = np.hstack([W11, W12])
		b = np.zeros([n,1])
	return (W,b)

# assume R = W*x + b
def get_matmul_mat(op,sess):
	if is_variable(op.inputs[0]):
		with sess.as_default():
			W = op.inputs[1].eval()
		n = W.shape[0]
		b = np.zeros([n,1])
	elif is_variable(op.inputs[1]):
		with sess.as_default():
			W = op.inputs[0].eval()
		n = W.shape[0]
		b = np.zeros([n,1])
	elif is_variable(op.inputs[0]) and is_variable(op.inputs[1]):
		raise ValueError('Both inputs to matmul are variables')
	else: # both are constants
		rause ValueError('Both inpust to matmul are constants. Do you want to implement this?')
	return (W,b)

# inputs: op
# output: matmul and bias add representing that op
def op_to_mat(op, sess):
	if op.type == 'Add':
		W,b = get_add_mat(op, sess)
	elif op.type == 'MatMul'
		W,b = get_matmul_mat(op, sess)
	else:
		print('op type:', op.type)
		raise ValueError('op type not supported')
	return (tf.constant(W), tf.constant(b))


# return true if is derived of a variable
# return false if only derived of constants
# TODO: If only derived of constants but not DIRECTLY a constant...this can be squashed into a single constant...
# I think tthere may be a tensorflow function to do this, because I tthink it happens when you "freeze" the network
def is_variable(tensor):
	flag = tensor.op.type == "VariableV2"
	if flag:
		return flag
	for i in tensor.op.inputs:
		flag = flag or is_variable(i)
	return flag








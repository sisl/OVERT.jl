# parsing code
# right now, assume that we DONT take batches. we only take one "datapoint" at a time
# assume multiplication of the form: W*x + b
import numpy as np
import tensorflow as tf

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


def are_inputs_vars(ops):	
	return [[is_variable(i) for i in op.inputs] for op in ops]

# stack matrices for different inputs
def matrix_stacker(mats):
	height = sum([m[0].shape[0] for m in mats])
	width = sum([m[0].shape[1] for m in mats])
	# for stacking mats, sttart with zero matrix
	# replace values fo specific pieces
	mega_mat = np.zeros((height, width))
	mega_bias = np.zeros((height, 1))
	i = 0
	j = 0
	for mat in mats:
		n = mat[0].shape[0]
		m = mat[0].shape[1]
		mega_mat[i:i+n,j:j+m] = mat[0]
		mega_bias[i:i+n] = mat[1]
		i=n
		j=m
	return mega_mat, mega_bias		

# inputs: final op
# outputs: list of weights and biases for a flattened representation of the network 
def parse_network(ops, temp_W, temp_b, final_W, final_b, activation_type, sess):
	if all([op.type not in [activation_type, 'Identity'] for op in ops]):
		mats = [op_to_mat(op, sess) for op in ops]
		# the line that is actually doing something interesting:
		###########################################
		mega_mat, mega_bias = matrix_stacker(mats)
		###########################################
		temp_W.append(mega_mat)
		temp_b.append(mega_bias)
		# get input ops to recurse on
		all_inputs = [op.inputs for op in ops]
		iops = [item.op for sublist in all_inputs for item in sublist]
		iops_inds = [is_variable(item) for sublist in all_inputs for item in sublist]
		var_iops = []
		# true-false indexing to get input ops that correspond to variables
		for i in range(len(iops)):
			if iops_inds[i]:
				var_iops.append(iops[i])
		## HANDLE DUPLICATES
		return parse_network(var_iops, temp_W, temp_b, final_W, final_b, activation_type, sess)

	elif all([op.type=='Identity' for op in ops]): 
		# all ops are of type "variable read"
		# this means we've gotten back to the beginning
		## HANDLE DUPLICATES
		if len(temp_W) > 0:
			W, b = condense_list(temp_W, temp_b)
			final_W.append(W)
			final_b.append(b)
		return final_W, final_b ## woo!!!! :D
	elif all([op.type==activation_type for op in ops]): 
		# if all ops are of type activation
		# multiply temporary tensor list togeter to produce final tensor. add the final tensor list
		# record activation type? into activation list?
		# potentially record into .nnet file?
		# recurse on input
		# BUT if temp list is empty, this means we are just STARTING a squish-tensor-phase
		pass
	else: # there is a mixture of real ops, variable reads, and/or activations
		# THENN add in "identity" ops for the variable reads and the activations until the real ops are "drawn down"
		# TODO next: write the above elif
		import pdb; pdb.set_trace()
		mats = []
		for op in ops:
			if op.type not in ['Identity', activation_type]:
				mats.append(op_to_mat(op,sess))
			else: 
				mats.append(get_identity_mat(op, activation_type))
		mega_mat, mega_bias = matrix_stacker(mats)
		temp_W.append(mega_mat)
		temp_b.append(mega_bias)
		# get inputs to recurse on (but not for activations and variable reads)
		var_iops = []
		for op in ops:
			if op.type not in ['Identity', activation_type]:
				for oi in op.inputs:
					if is_variable(oi):
						var_iops.append(oi.op)
			else:
				var_iops.append(op)
		## HANDLE DUPLICATES
		return parse_network(var_iops, temp_W, temp_b, final_W, final_b, activation_type, sess)

# usage:
# s = set(op_inputs)
# if len(s) < len(op_inputs):
# 		W,b = handle_duplicates(op_inputs)
# 		temp_W.append(W)
# 		temp_b.append(b)
# # ELSE: continue as normal
def handle_duplicates(op_inputs):
	import pdb; pdb.set_trace()
	d = {}
	letter = 'A'
	for oi in op_inputs:
		if oi in d.keys():
			pass
		else: 
			d[oi] = letter
			letter = get_next_letter(letter)
	# ending point:
	labeled_ois = [d[oi] for oi in op_inputs]
	print("labeled_inputs: ", labeled_ois)
	# starting point
	# create unique list of inputs
	s = set(op_inputs)
	n = len(s)
	unique_ois = [s.pop() for i in range(n)]
	unique_labeled_ois = [d[oi] for oi in unique_ois]
	print("set of inputs: ", unique_labeled_ois)
	# create matrix taking us from starting point to ending point
	m = np.chararray((len(op_inputs), n))
	m[:]='0'
	# write one row at a time
	# the index where the 'I' goes is the index in the set ("starting point") of the character in the ending point
	rows = len(op_inputs)
	for r in range(rows):
		ind = unique_labeled_ois.index(labeled_ois[r])
		m[r,ind] = 'I'
	print("conversion matrix: ", m)

	# now convert back to numbers
	
	return m

def get_next_letter(l):
	if l[-1] == 'Z':
		new_l = 'A'*(len(l)+1)
	else:
		n = ord(l[-1])
		new_l = l[0:-1]+chr(n+1) 
	return new_l

def get_identity_mat(op, activation_type):
	n = op.outputs[0].shape[0].value
	W = np.eye(n)
	b = np.zeros([n,1])
	return W,b

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
		raise ValueError('Both inputs to matmul are constants. Do you want to implement this?')
	return (W,b)

# inputs: op
# output: matmul and bias add representing that op
def op_to_mat(op, sess):
	if op.type == 'Add':
		W,b = get_add_mat(op, sess)
	elif op.type == 'MatMul':
		W,b = get_matmul_mat(op, sess)
	else:
		print('op type:', op.type)
		raise ValueError('op type not supported')
	return W,b


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








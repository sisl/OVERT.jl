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
		import pdb; pdb.set_trace()
		var_inputs, var_iops = get_inputs(ops)
		## HANDLE DUPLICATES (also an interesting line)
		s = set(var_inputs)
		if len(s) < len(var_inputs):
			W,b, unique_vis = handle_duplicates(var_inputs)
			temp_W.append(W)
			temp_b.append(b)
			var_iops = [uv.op for uv in unique_vis]
		# ELSE: continue as normal
		return parse_network(var_iops, temp_W, temp_b, final_W, final_b, activation_type, sess)

	elif all([op.type=='Identity' for op in ops]): 
		# all ops are of type "variable read"
		# this means we've gotten back to the beginning
		## HANDLE DUPLICATES
		# I think it's possible we'll never "NEED" to handle duplicates here because they were handled BEFORE we got passed here...
		signals = []
		for op in ops:
			for oo in op.outputs:
				signals.append(oo)
		s = set(signals)
		if len(s) < len(signals):
			W,b, unique_s = handle_duplicates(signals)
			temp_W.append(W)
			temp_b.append(b)
		import pdb; pdb.set_trace()
		# squish time
		if len(temp_W) > 0:
			W, b = condense_list(temp_W, temp_b)
			final_W.append(W)
			final_b.append(b)
		return final_W, final_b ## woo!!!! :D
	elif all([op.type==activation_type for op in ops]): 
		# if all ops are of type activation
		# multiply temporary tensor list together to produce final tensor. add to the final tensor list and then empty temp tensor lists
		# record activation type? into activation list?
		# potentially record into .nnet file?
		# recurse on input 
		# assume all activations are the same FOR NOW
		import pdb; pdb.set_trace()
		# squish time
		if len(temp_W) > 0:
			W, b = condense_list(temp_W, temp_b)
			final_W.append(W)
			final_b.append(b)
			temp_W = []
			temp_b = []

		# get inputs to recurse on
		var_inputs, var_iops = get_inputs(ops)
		## HANDLE DUPLICATES (also an interesting line)
		s = set(var_inputs)
		if len(s) < len(var_inputs):
			W,b, unique_vis = handle_duplicates(var_inputs)
			temp_W.append(W)
			temp_b.append(b)
			var_iops = [uv.op for uv in unique_vis]
		# ELSE: continue as normal
		return parse_network(var_iops, temp_W, temp_b, final_W, final_b, activation_type, sess)

	else: # there is a mixture of real ops, variable reads, and/or activations
		# THENN add in "identity" ops for the variable reads and the activations until the real ops are "drawn down"
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
		var_inputs = []
		for op in ops:
			if op.type not in ['Identity', activation_type]:
				for oi in op.inputs:
					if is_variable(oi):
						var_iops.append(oi.op)
						var_inputs.append(oi)
			else:
				var_iops.append(op)
				for oo in op.outputs:
					var_inputs.append(oo)
		## HANDLE DUPLICATES
		s = set(var_inputs)
		if len(s) < len(var_inputs):
			W,b, unique_vis = handle_duplicates(var_inputs)
			temp_W.append(W)
			temp_b.append(b)
			var_iops = [uv.op for uv in unique_vis]
		return parse_network(var_iops, temp_W, temp_b, final_W, final_b, activation_type, sess)

def get_inputs(ops):
	all_inputs = [op.inputs for op in ops]
	var_inputs = []
	var_iops = []
	# true-false indexing to get input ops that correspond to variables
	for sublist in all_inputs:
		for item in sublist:
			if is_variable(item):
				var_inputs.append(item)
				var_iops.append(item.op)
	return var_inputs, var_iops

def condense_list(W_list, b_list):


# usage:
# s = set(op_inputs)
# if len(s) < len(op_inputs):
# 		W,b, unique_ois = handle_duplicates(op_inputs)
# 		temp_W.append(W)
# 		temp_b.append(b)
# # ELSE: continue as normal
# 
# A function to handle 'splits' that left unhandled would lead to duplicate signals
# CHECK: only variable inputs are passed to this function?
def handle_duplicates(op_inputs):
	# ending point: op_inputs
	# starting point
	n_orig = len(op_inputs)
	# create unique list of inputs
	s = set(op_inputs)
	n = len(s)
	unique_ois = [s.pop() for i in range(n)]
	#print("unique_ois: ", unique_ois)
	# create matrix taking us from starting point to ending point
	m = np.zeros((n_orig, n))
	# write one row at a time
	# the index where the 'I' goes is the index in the set ("starting point") of the character in the ending point
	rows = n_orig
	for r in range(rows):
		ind = unique_ois.index(op_inputs[r])
		m[r,ind] = 1
	print("conversion matrix: ", m)
	# now convert back to real dimensions
	# width (of megamat): sum of widths of identity matrices used for each element in set, which is also height of the corresponding input
	width = sum([o.shape[0].value for o in unique_ois])
	# height (of megamat): sum of heights of op_inputs 
	height = sum([o.shape[0].value for o in op_inputs])
	mat = np.zeros((height, width))
	# create corresponding list of identity matrices for unique inputs 
	eyes = []
	widths = []
	for uo in unique_ois:
		eyes.append(np.eye(uo.shape[0].value))
		widths.append(eyes[-1].shape[1]) 
	# create lists of indices where columns start and end
	c_starts = [0]
	c_starts.extend(np.cumsum(widths[0:-1]))
	c_ends =c_starts[1:]
	c_ends.append(width)
	assert(len(c_starts) == len(c_ends))
	cols = n
	r_start = 0
	for r in range(rows):
		for c in range(cols):
			if m[r,c] == 1:
				end_r = r_start + eyes[c].shape[0]
				c_start = c_starts[c]
				c_end = c_ends[c]
				mat[r_start:end_r, c_start:c_end] = eyes[c]
		r_start = end_r
	b = np.zeros((height, 1))
	return mat, b, unique_ois

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
def is_variable(tensor):
	flag = tensor.op.type == "VariableV2"
	if flag:
		return flag
	for i in tensor.op.inputs:
		flag = flag or is_variable(i)
	return flag








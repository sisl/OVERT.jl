# parsing code

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
		input_varflags = are_inputs_vars(op)
		# if op has more than 1 variable input:
		if sum(input_varflags) > 1: 
			# get input ops
			iops = [oi.op for oi in op.inputs]
			iop_varflags = [sum(are_inputs_vars(o)) for o in iops]
			# if input ops just have 1 variable input each
			if len(iops) == sum(iop_varflags):
				# restack those two input ops into 1
				restacked = matrix_stacker(iops)
				# return: continue to recurse on inputs to newly stacked op
				return parse_network()
			# else if input ops have more than 1 variable input each...
				# recurse on each input op with a "blank" tensor/matrix list
				# restack input ops into a single op
				# continue to recurse on inputs to newly stacked op
		else # if op just has 1 variable input:
			# include tensor in list
			# recurse on input
	else # if op is of type activation
		# multiply temporary tensor list togeter to produce final tensor. add the final tensor list
		# record activation type? into activation list?
		# potentially record into .nnet file?
		# recurse on input

# return true if is derived of a variable
# return false if only derived of constants
# TODO: If only derived of constants but not DIRECTLY a constant...this can be squashed into a single constant...
# I think tthere may be a tensorflow function to do this, because I tthink it happens when you "freeze" tthe network
def is_variable(tensor):
	flag = tensor.op.type == "VariableV2"
	if flag:
		return flag
	for i in tensor.op.inputs:
		flag = flag or is_variable(i)
	return flag

# inputs: matrices
# output: one signle stacked matrix
def matrix_stacker(?):








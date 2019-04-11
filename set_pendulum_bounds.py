from maraboupy.MarabouUtils import *
import numpy as np
import os

# turn ordering of inputs into dicts
# dict 1: long inputs -> short dict key names (e.g. theta_dot_hats)
# dict 2: long_inputs -> variable numbers
def map_inputs(network, inputs):
    n_in = network.inputVars[0]
    d1 = dict()
    d2 = dict()
    for i in range(len(inputs)):
        str_i = inputs[i] 
        if "theta_dot_hat" in str_i:
            d1[str_i] = "theta_dot_hats"
        elif "theta_dot_0" in str_i:
            d1[str_i] = "theta_dot_0"
        elif "theta_0" in str_i:
            d1[str_i] = "theta_0"
        d2[str_i] = n_in[i][0]
    return d1, d2    

def map_inputs_fromVarMap(varMap, input_ops):
    d1 = dict()
    d2 = dict()
    for i in range(len(input_ops)):
        str_i = input_ops[i] # truly names of input ops 
        if "theta_dot_hat" in str_i:
            d1[str_i] = "theta_dot_hats"
        elif "theta_dot_0" in str_i:
            d1[str_i] = "theta_dot_0"
        elif "theta_0" in str_i:
            d1[str_i] = "theta_0"
        d2[str_i] = varMap[str_i][0][0]
    return d1, d2        
# use these two dictionaries like:
#upperBound(dict2[long_input] 
#   = input_max_value[dict1[long_input]])

def log_bounds(fdir, run_n, bounds):
    fname = os.path.join(fdir, "bounds_"+run_n+".txt")
    with open(fname,'w') as f:
        f.write("Input mins \n")
        f.write(str(bounds.inputs_min))
        f.write("Input maxes \n")
        f.write(str(bounds.inputs_max))
        f.write("Output mins \n")
        f.write(str(bounds.outputs_min))
        f.write("Output maxes \n")
        f.write(str(bounds.outputs_max))



# dict 1: long inputs -> short dict key names (e.g. theta_dot_hats)
# dict 2: long_inputs -> variable numbers
def set_bounds(network, input_long2short, input_long2var, bounds_fun, fdir, run_n):
    # get order of inputs in ffnetwork
    il2s = input_long2short
    il2v = input_long2var

    # get specific bound values
    bounds = bounds_fun()
    log_bounds(fdir, run_n, bounds)
    # set input bounds
    for li in il2s.keys():
        if il2s[li] in bounds.inputs_min:
            network.setLowerBound(il2v[li], bounds.inputs_min[il2s[li]])
        else:
            print(il2s[li], " not in bounds object min")
        if il2s[li] in bounds.inputs_max:
            network.setUpperBound(il2v[li], bounds.inputs_max[il2s[li]])
        else:
            print(il2s[li], " not in bounds object max")

    # set output bounds bc they'll always be in the same order
    # first three are thetas, 3 tdh, 3 tdlb, 3 tdub
    # 0,1,2    3,4,5    6,7,8   9,10,11
    # i = 0,1,2
    # j = i+3
    output_vars_shape = np.array(network.outputVars).shape
    output_vars = np.array(network.outputVars).flatten()
    print("original output vars: ", output_vars)

    nsteps = int(len(output_vars) / 4.0) # e.g. 3
    for i in range(nsteps): # only bound thetas
        # set hyperrectangle bounds
        # bounds we care about
        if "thetas" in bounds.outputs_min:
            LB = bounds.outputs_min["thetas"]
            UB = bounds.outputs_max["thetas"]
            Y = addComplementOutputSet(network, 
                LB=LB, 
                UB=UB,
                x=output_vars[i]
                )
        elif "theta_"+str(i+1) in bounds.outputs_min:
            print("Applying complement output bound for ", "theta_"+str(i+1))
            LB = bounds.outputs_min["theta_"+str(i+1)]
            UB = bounds.outputs_max["theta_"+str(i+1)]
            Y = addComplementOutputSet(network, 
                LB=LB, 
                UB=UB,
                x=output_vars[i]
                )
        # bounds we care about:
        # set inequality bounds: vars_i*coeffs_i <= scalar
        # LB - tdh <=0   -->  LB <= tdh
        #import pdb; pdb.set_trace()
        addInequality(network, [output_vars[i+2*nsteps], output_vars[i+nsteps]], [1.0, -1.0], 0.0) 
        # -UB + tdh <= 0 -->  tdh <= UB
        addInequality(network, [output_vars[i+3*nsteps], output_vars[i+nsteps]], [-1.0, 1.0], 0.0) 
        # result: LB <= tdh <= UB

    network.outputVars = output_vars[:nsteps].reshape(nsteps,1)
    # outputs are thetas only

    return bounds


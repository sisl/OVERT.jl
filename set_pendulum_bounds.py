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

def log_bounds(fdir, run_n, inputs_min, inputs_max, outputs_min, outputs_max):
    fname = os.path.join(fdir, "bounds_"+run_n+".txt")
    with open(fname,'w') as f:
        f.write("Input mins \n")
        f.write(str(inputs_min))
        f.write("Input maxes \n")
        f.write(str(inputs_max))
        f.write("Output mins \n")
        f.write(str(outputs_min))
        f.write("Output maxes \n")
        f.write(str(outputs_max))



# dict 1: long inputs -> short dict key names (e.g. theta_dot_hats)
# dict 2: long_inputs -> variable numbers
def set_bounds(network, input_long2short, input_long2var, bounds_fun, fdir, run_n):
    # get order of inputs in ffnetwork
    il2s = input_long2short
    il2v = input_long2var

    # get network input vars
    input_vars = network.inputVars[0]

    # get specific bound values
    inputs_min, inputs_max, outputs_min, outputs_max = bounds_fun()
    log_bounds(fdir, run_n, inputs_min, inputs_max, outputs_min, outputs_max)
    # set input bounds
    for li in il2s.keys():
        network.setLowerBound(il2v[li], inputs_min[il2s[li]])
        network.setUpperBound(il2v[li], inputs_max[il2s[li]])
    # network.setLowerBound(input_vars[input_dict["theta_0"]][0], inputs_min["theta_0"])
    # network.setUpperBound(input_vars[input_dict["theta_0"]][0], inputs_max["theta_0"])
    # print("theta0",input_vars[input_dict["theta_0"]][0])
    # network.setLowerBound(input_vars[input_dict["theta_dot_0"]][0], inputs_min["theta_dot_0"])
    # network.setUpperBound(input_vars[input_dict["theta_dot_0"]][0], inputs_max["theta_dot_0"])
    # print("thetadot0", input_vars[input_dict["theta_dot_0"]][0])
    # ntdhs = len(input_vars) - 2
    # for i in range(ntdhs):
    #     network.setLowerBound(input_vars[input_dict["theta_dot_hats"][i]][0], inputs_min["theta_dot_hats"])
    #     network.setUpperBound(input_vars[input_dict["theta_dot_hats"][i]][0], inputs_max["theta_dot_hats"])
    #     print("theta dot hats:", input_vars[input_dict["theta_dot_hats"][i]][0])

    # set output bounds bc they'll always be in the same order
    # first three are thetas, 3 tdh, 3 tdlb, 3 tdub
    # 0,1,2    3,4,5    6,7,8   9,10,11
    # i = 0,1,2
    # j = i+3
    output_vars = network.outputVars
    network.outputVars = None
    nsteps = int(len(output_vars) / 4.0) # e.g. 3
    for i in range(nsteps):
        # set hyperrectangle bounds
        # bounds we care about
        addComplementOutputSet(network, 
            LB=outputs_min["thetas"], 
            UB=outputs_max["thetas"],
            x=output_vars[i][0]
            )
        # bounds we have to set
        addComplementOutputSet(network, 
            LB=outputs_min["theta_dot_hats"], 
            UB=outputs_max["theta_dot_hats"],
            x=output_vars[i+nsteps][0]
            )
        addComplementOutputSet(network, 
            LB=outputs_min["tdlbs"], 
            UB=outputs_max["tdlbs"],
            x=output_vars[i+2*nsteps][0]
            )
        addComplementOutputSet(network, 
            LB=outputs_min["tdubs"], 
            UB=outputs_max["tdubs"],
            x=output_vars[i+3*nsteps][0]
            )
        # set inequality bounds: vars_i*coeffs_i <= scalar
        addInequality(network, [output_vars[i+6][0], output_vars[i+3][0]], [1.0, -1.0], 0.0, "LE") # theta dot greater than LB
        addInequality(network, [output_vars[i+9][0], output_vars[i+3][0]], [-1.0, -1.0], 0.0, "LE") # theta dot less than UB










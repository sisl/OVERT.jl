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


# dict 1: long inputs -> short dict key names (e.g. theta_dot_hats)
# dict 2: long_inputs -> variable numbers
def set_bounds(network, input_long2short, input_long2var, bounds_fun, fdir, run_n):
    # get order of inputs in ffnetwork
    il2s = input_long2short
    il2v = input_long2var

    # log specific bound values
    bounds = bounds_fun()
    bounds.log(fdir, run_n)
    # set input bounds
    import pdb; pdb.set_trace()
    for li in il2s.keys():
        if il2s[li] in bounds.inputs_min:
            network.setLowerBound(il2v[li], bounds.inputs_min[il2s[li]])
        else:
            print(il2s[li], " not in bounds object min")
        if il2s[li] in bounds.inputs_max:
            network.setUpperBound(il2v[li], bounds.inputs_max[il2s[li]])
        else:
            print(il2s[li], " not in bounds object max")

    # set output bounds -- they'll always be in the same order
    # first three are thetas, 3 tdh, 3 tdlb, 3 tdub
    # 0,1,2    3,4,5    6,7,8   9,10,11
    # i = 0,1,2
    # j = i+3
    output_vars_shape = np.array(network.outputVars).shape
    output_vars = np.array(network.outputVars).flatten()
    print("original output vars: ", output_vars)

    nsteps = int(len(output_vars) / 4.0) # e.g. 3
    print("nsteps: ", nsteps)
    # for all complement_output_sets that we add, we want to OR all of them
    complement_output_sets = []
    for i in range(nsteps):
        # set hyperrectangle output bounds
        # theta output bounds
        theta_i = "theta_"+str(i+1)
        if theta_i in bounds.outputs_min:
            print("Applying complement output bound for ", theta_i)
            LB = bounds.outputs_min[theta_i]
            UB = bounds.outputs_max[theta_i]
            complement_output_sets.append((LB, UB, output_vars[i]))
        # theta dot hat output bounds
        theta_dot_i = "theta_dot_"+str(i+1)
        if theta_dot_i in bounds.outputs_min:
            print("Applying complement output bound for ", theta_dot_i )
            LB = bounds.outputs_min[theta_dot_i]
            UB = bounds.outputs_max[theta_dot_i]
            complement_output_sets.append((LB, UB, output_vars[i+nsteps]))
        # set inequality bounds as a constraint, not a check: vars_i*coeffs_i <= scalar
        # LB - tdh <=0   -->  LB <= tdh
        #import pdb; pdb.set_trace()
        addInequality(network, [output_vars[i+2*nsteps], output_vars[i+nsteps]], [1.0, -1.0], 0.0) 
        # -UB + tdh <= 0 -->  tdh <= UB
        addInequality(network, [output_vars[i+3*nsteps], output_vars[i+nsteps]], [-1.0, 1.0], 0.0) 
        # result: LB <= tdh <= UB

    # actually take all planned complement output set bounds and implement them
    # with 1 big "OR" statement
    success = addComplementOutputSets(network, complement_output_sets)

    #network.outputVars = output_vars[:nsteps].reshape(nsteps,1)
    # outputs are thetas only

    return bounds


# a function where you assume bounds 0:k-1, test bounds at step k, and leave bounds
# for steps k:n unset
# dict 1: long inputs -> short dict key names (e.g. theta_dot_hats)
# dict 2: long_inputs -> variable numbers
def set_k_bounds(network, input_long2short, input_long2var, bounds_fun, fdir, run_n, stepk):
    # get order of inputs in ffnetwork
    il2s = input_long2short
    il2v = input_long2var

    # log specific bound values
    bounds = bounds_fun()
    bounds.log(fdir, run_n)
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

    # set output bounds -- they'll always be in the same order
    # first three are thetas, 3 tdh, 3 tdlb, 3 tdub
    # 0,1,2    3,4,5    6,7,8   9,10,11
    # i = 0,1,2
    # j = i+3
    output_vars_shape = np.array(network.outputVars).shape
    output_vars = np.array(network.outputVars).flatten()
    print("original output vars: ", output_vars)

    nsteps = int(len(output_vars) / 4.0) # e.g. 3
    print("testing bounds for step k:", stepk)
    print("total nsteps: ", nsteps)

    # for all complement_output_sets that we add, we want to OR all of them
    complement_output_sets = []
    for i in range(nsteps):
        theta_i = "theta_" + str(i + 1)
        theta_dot_i = "theta_dot_" + str(i + 1)
        if i in range(1,stepk):
            ########################################
            # set "assume" bounds for steps 1:k-1
            # for theta, set bounds
            if theta_i in bounds.outputs_min:
                LB = bounds.outputs_min[theta_i]
                UB = bounds.outputs_max[theta_i]
                network.setLowerBound(output_vars[i], LB)
                network.setUpperBound(output_vars[i], UB)
                print("apply assume bound for ", theta_i)
            if theta_dot_i in bounds.outputs_min:
                LB = bounds.outputs_min[theta_dot_i]
                UB = bounds.outputs_max[theta_dot_i]
                network.setLowerBound(output_vars[i + nsteps], LB)
                network.setUpperBound(output_vars[i + nsteps], UB)
                print("apply assume bound for ", theta_dot_i)
            # set inequality bounds as a constraint, not a check: vars_i*coeffs_i <= scalar
            # tdhLB - tdh <=0   -->  tdhLB <= tdh
            addInequality(network, [output_vars[i + 2 * nsteps], output_vars[i + nsteps]], [1.0, -1.0], 0.0)
            # -tdhUB + tdh <= 0 -->  tdh <= tdhUB
            addInequality(network, [output_vars[i + 3 * nsteps], output_vars[i + nsteps]], [-1.0, 1.0], 0.0)
            # result: tdhLB <= tdh <= tdhUB
        elif i == stepk:
            ########################################
            # set complement output set bounds for k
            # set hyperrectangle output "test" bounds
            # theta output bounds
            if theta_i in bounds.outputs_min:
                print("Applying complement output bound for ", theta_i)
                LB = bounds.outputs_min[theta_i]
                UB = bounds.outputs_max[theta_i]
                complement_output_sets.append((LB, UB, output_vars[i]))
            # theta dot hat output bounds
            if theta_dot_i in bounds.outputs_min:
                print("Applying complement output bound for ", theta_dot_i )
                LB = bounds.outputs_min[theta_dot_i]
                UB = bounds.outputs_max[theta_dot_i]
                complement_output_sets.append((LB, UB, output_vars[i+nsteps]))
            # set inequality bounds as a constraint, not a check: vars_i*coeffs_i <= scalar
            # tdhLB - tdh <=0   -->  tdhLB <= tdh
            addInequality(network, [output_vars[i + 2 * nsteps], output_vars[i + nsteps]], [1.0, -1.0], 0.0)
            # -tdhUB + tdh <= 0 -->  tdh <= tdhUB
            addInequality(network, [output_vars[i + 3 * nsteps], output_vars[i + nsteps]], [-1.0, 1.0], 0.0)
            # result: tdhLB <= tdh <= tdhUB
        else: # i in range stepk + 1 : nsteps
            ########################################
            # leave bounds k+1:n unset
            pass

    # take all planned complement output set bounds and implement them
    # with 1 big "OR" statement
    success = addComplementOutputSets(network, complement_output_sets)

    return bounds
# using marabou utils
import colored_traceback.always
import numpy as np
import tensorflow as tf
from maraboupy.MarabouUtils import *
import gym
import matplotlib.pyplot as plt
import os
from simple_wrapper import SimpleWrapper
from maraboupy import Marabou
import sys


def get_adjustable_vars(adj_inputs, d2):
    # d1 maps long input names to variable numbers
    adjustable_vars = []
    for i in adj_inputs:
        adjustable_vars.append(d2[i])
    return adjustable_vars

def get_dependencies(nsteps):
    dependencies = []
    for i in range(nsteps):
        if i == 0:
            pfx1 = ""
        else:
            pfx1 = "_"+str(i)
        pfx2 = str(i+1)
        s1 = "run_dynamics"+pfx1+"/Dynamics_"+pfx2+"/thdotLB/theta_dot_LB_"+pfx2+":0"
        s2 = "run_dynamics"+pfx1+"/Dynamics_"+pfx2+"/thdotUB/theta_dot_UB_"+pfx2+":0"
        d = "assign_init_vals/theta_dot_hat_"+pfx2+":0"
        dependencies.append( ((s1,s2),d) )
    return dependencies

def read_inout_metadata(meta_data):
    with open(meta_data) as f:
        line = f.readline()
        output_op_name = f.readline()[0:-1] # should contain name of output op for ffnetwork
        line = f.readline()
        inputs = f.readline().split(',')[0:-1] # discard newline character at end of line
        line = f.readline()
        outputs = f.readline()
        return output_op_name, inputs, outputs

def read_extra_mass_metadata(file):
    with open(file) as f:
        line = f.readline()
        i = line.find(":")
        nsteps = int(line[i+1:])
        # find m1
        line = f.readline()
        i = line.find("=")
        m1 = float(line[i+1:])
        # find m2
        line = f.readline()
        i = line.find("=")
        m2 = float(line[i+1:])
        # find l
        line = f.readline()
        i = line.find("=")
        l = float(line[i+1:])
    return nsteps, m1, m2, l


def read_output_metadata(meta_data):
    with open(meta_data) as f:
        line = f.readline()
        output_op_name = f.readline()[0:-1] # should contain name of output op
        line = f.readline()
        outputs = f.readline().split(',')[0:-1] # discard newline character at end of line
        return output_op_name, outputs

def varMapOpstoNames(varMap):
    """
    Take a dictionary that maps op objects to marabou variable numbers and return
    a dictionary that maps op NAMES to marabou variable numbers
    """
    varMapNames = dict()
    for k in varMap.keys():
        varMapNames[k.name] = varMap[k]
    return varMapNames 

def display_equations(network):
    zipped_eqs = zip([e.addendList for e in network.equList], [e.scalar for e in network.equList])
    print("Equations: ",[z for z in zipped_eqs])
    if len(network.maxList)>0:
        print("Max List: ", network.maxList)


def check_bounds(upper, lower):
    """
    make sure all lower <= upper
    """
    for k in upper.keys():
        if k in lower:
            assert lower[k] <= upper[k]

    for k in lower.keys():
        if k in upper:
            assert lower[k] <= upper[k]
    print("bounds check passed")
    return True


def get_specific_bounds(variables, bounds):
    bound_val_dict = {}
    for v in np.array(variables).flatten():
        if v in bounds:
            bound_val_dict[v] = bounds[v]
    return bound_val_dict


# find equations with specific variables in them
def find_spec_equ(equList, varss):
    equns = []
    for e in equList:
        for v in varss:
            if v in [t[1] for t in e.addendList]:
                equns.append(stringify(e))
    return equns


# turn equations into nice strings
def stringify(eq):
    s = ""
    for item in eq.addendList:
        s = s + str(item[0]) + "*v" + str(item[1]) + " + "
    s = s[:-2] + "<= " + str(eq.scalar)
    return s

def eval_marabou(network, theta, theta_dot):
    # eval with marabou
    network.setLowerBound(0, theta)
    network.setUpperBound(0, theta)
    network.setLowerBound(1, theta_dot)
    network.setUpperBound(1, theta_dot)
    print_io_bounds(network, network.inputVars, network.outputVars)
    vals, stats, exit_code = network.solve()
    return (vals, stats, exit_code)

def read_graph(filename):
    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    sess = tf.Session(graph=graph)
    return sess

def eval_from_tf(filename, input_dict, outputs):
    # filename: string of .pb filename with frozen graph
    # input_dict: (opname, value) pairs in a dictionary
    # outputs: opname array-like
    sess = read_graph(filename)
    feed_dict = {}
    for k in input_dict.keys():
        feed_dict[k+":0"] = input_dict[k]
    outputs = [o+":0" for o in outputs]
    output_vals = sess.run(outputs, feed_dict = feed_dict)
    return output_vals

def compare_marabou_tf(network, theta, theta_dot, nsteps, outputVarList):
    # evaluate and make sure that marabou and tensorflow produce the same outputs
    # evlaute with maraboou: fine, outputs will be thetas
    # evaluate without marabou (with TF), outputs will be the single concatenated output
    # of thetas, theta-dot-hats, thdlbs, and thdubs

    # inequality bounds:
    output_vars = np.array(network.outputVars).flatten()
    print("original output vars: ", output_vars)
    # import pdb; pdb.set_trace()
    #network.outputVars = None
    nsteps = int(len(output_vars) / 4.0) # e.g. 3
    for i in range(nsteps):
        # set inequality bounds: vars_i*coeffs_i <= scalar
        # LB - tdh <=0   -->  LB <= tdh
        addInequality(network, [output_vars[i+2*nsteps], output_vars[i+nsteps]], [1.0, -1.0], 0.0) 
        # -UB + tdh <= 0 -->  tdh <= UB
        addInequality(network, [output_vars[i+3*nsteps], output_vars[i+nsteps]], [-1.0, 1.0], 0.0) 

    # NOTE: make sure you haven't set input bounds on theta...OR output bounds....
    # eval with marabou
    inputs = [theta, theta_dot]
    for i in range(len(inputs)):
        network.setLowerBound(i, inputs[i])
        network.setUpperBound(i, inputs[i])
    print_io_bounds(network, network.inputVars, network.outputVars)
    vals, stats, exit_code = network.solve()

    for i in range(nsteps):
        inputs.append(vals[2+i]) # add theta dot hat values

    # eval with TF
    tf_inputs = np.array(inputs).reshape(2+nsteps, 1)
    outputs_tf = network.evaluateWithoutMarabou(tf_inputs)

    # ensure values from each evaluation method are the same
    for i in range(len(outputVarList)):
        print("tf output: ",outputs_tf.flatten()[i])
        print("marabou output: ", vals[outputVarList[i]])
        assert(abs(outputs_tf.flatten()[i] - vals[outputVarList[i]])<1e-5) # assert all values very close
    print("Values from each method are very close!")
    # remove fixed input values for upper and lower bounds
    # on theta0 and theta_dot_0
    network.lowerBounds.pop(0)
    network.upperBounds.pop(0)
    network.lowerBounds.pop(1)
    network.upperBounds.pop(1)


def print_io_bounds(network, inputVars, outputVars):
    input_ub = get_specific_bounds(inputVars, network.upperBounds)
    print("input_ub: ", input_ub)
    input_lb = get_specific_bounds(inputVars, network.lowerBounds)
    print("input_lb: ", input_lb)
    output_ub = get_specific_bounds(outputVars, network.upperBounds)
    print("output_ub: ", output_ub)
    output_lb = get_specific_bounds(outputVars, network.lowerBounds)
    print("output_lb: ", output_lb)

def plot_bounded_dynamics(filename, fixed_vars, var_to_sample, outputs, fun):
    # fixed vars: {var: val, var: val,...}
    # vars to sample: (var: [a,b])
    # outputs: [a1, a2, a3, ...]
    # fun: analytical form of the dynamics function
    # NOTE: untested.
    # the idea is to plot (a slice of) the bounds of the dynamics function and 
    # the function itself
    sess = read_graph(filename)
    feed_dict = fixed_vars
    k = var_to_sample[0]
    a = var_to_sample[1][0]
    b = var_to_sample[1][1]
    x = np.zeros((1000,1))
    y = np.zeros((1000,len(outputs)))
    z = np.linspace(a,b, 1000)
    q = np.zeros((1000,1))
    for i in range(1000):
        x[i] = (np.random.rand()*(b-a) + a)
        feed_dict[k+":0"] = x[i]
        output_vals = sess.run(outputs, feed_dict = feed_dict)
        y[i,:] = np.array(output_vals).flatten().reshape(1,-1)
        q[i] = fun(z[i])
    # plotting
    f = plt.figure()
    for i in range(len(outputs)):
        plt.plot(x, y[:,i])
    plt.plot(z,q, label="original")    
    plt.show()
    return f

def load_network(frozen_graph):
    # load network
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
    return sess

def get_output(op_name, graph=tf.get_default_graph()):
    return graph.get_operation_by_name(op_name).outputs[0]

class Lightweight_policy():
    def __init__(self, sess):
        self.sess = sess
        self.controller_output = get_output("get_action/add_3", graph=sess.graph)
        # high = np.array([np.pi, np.inf])
        # self.observation_space = Box(low=-high, high=high)
    def reset(self):
        pass
    def get_action(self, state):
        state_t = state.flatten()
        #print("theta for getting action: ", state_t[0]*180/np.pi, " deg")
        action = self.sess.run([self.controller_output],
            feed_dict = {
            get_output("initial_values/theta_0", graph=self.sess.graph): 
                np.array([[state_t[0]]]),
            get_output("initial_values/theta_dot_0", graph=self.sess.graph): 
            np.array([[state_t[1]]])
            })
        return action[0], {"":""}

def stepEnv(env, policy, nsteps, vals, render, verbose=True):
    # step through nsteps steps of simulation
    if verbose:
        print("begin sim.")
    env.reset()
    env.set_state(np.array([vals[0], vals[1]])) # set theta0 and thetadot0
    if render:
        env.render()
    obsList = []
    for i in range(nsteps):
        action, _ = policy.get_action(env.get_state())
        obs, _, _, _ = env.step(action) # get theta and thetadot
        obsList.append(obs)
        if render:
            env.render()
            input("enter to continue")
    if verbose:
        print("end sim.")
    return obsList # thetas and thetadots

def check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps, render=True, verbose=True):
    # check if SAT example is REAL or due to OVERAPPROX
    # gonna wanna import mypendulumenv
    # pull the controller out of the composed graph
    # SET the state of the pendulum using the sat values
    # step twice 
    # look at the state values and compare them to limits
    # then announce whether TRUE SAT or OVERAPPROX SAT 
    policy = Lightweight_policy(sess)
    obsList = stepEnv(env, policy, nsteps, vals, render, verbose=verbose)
    # compare values
    # compare theta values

    if verbose:
        print("theta_0: ", vals[0]*180/np.pi)
        print("theta_dot_0: ", vals[1]*180/np.pi)

    sov = nsteps + 2 # start output var
    inbounds = True
    for i in range(nsteps):
        tkey = "theta_"+str(i+1)
        a = obsList[i][0] <= bounds.outputs_max[tkey]
        b = obsList[i][0] >= bounds.outputs_min[tkey]
        e = vals[sov+i] < bounds.outputs_max[tkey]
        f =  vals[sov+i] >  bounds.outputs_min[tkey]
        if verbose:
            print(tkey, " min: ", bounds.outputs_min[tkey]*180/np.pi)
            print(tkey, " max: ", bounds.outputs_max[tkey]*180/np.pi)
            print(tkey," from sim: ", obsList[i][0]*180/np.pi) # theta1
            print(tkey," from SAT: ", vals[sov+i]*180/np.pi)
            if not (a and b):
                print(tkey, " failure")
            if not (e and f):
                print(tkey, " SAT failure")
        #
        tdkey = "theta_dot_"+str(i+1)
        c = obsList[i][1] <= bounds.outputs_max[tdkey]
        d = obsList[i][1] >= bounds.outputs_min[tdkey]
        g = (bounds.outputs_max[tdkey] - vals[sov+2*nsteps+i]) > 1e-5 
        h = (vals[sov+2*nsteps+i] - bounds.outputs_min[tdkey]) > 1e-5
        if verbose:
            print(tdkey, " min: ", bounds.outputs_min[tdkey]*180/np.pi)
            print(tdkey, " max: ", bounds.outputs_max[tdkey]*180/np.pi)
            print(tdkey, "_LB: ", vals[sov+2*nsteps+i]*180/np.pi)
            print(tdkey, "_UB: ", vals[sov+3*nsteps+i]*180/np.pi)
            # recall: LB and UB are calculated from SAT example, not from sim trajectory. and may disagree after 1st timestep
            print(tdkey, " from sim: ", obsList[i][1]*180/np.pi) # thetadot1
            print(tdkey, " from SAT: ", vals[sov+nsteps+i]*180/np.pi)
            if not (c and d):
                print(tdkey, " failure")
            if not (g and h):
                print(tdkey, " SAT failure")
            #
        inbounds = inbounds and a and b and c and d

    if inbounds:
        return "OVERAPPROX"
    else:
        return "REAL"

# vary mass and see if we can find a real counter example
def check_SAT_REAL_or_OVERAPPROX_varyM(sess, vals, env, bounds, nsteps, render=True, nsims=1000):
    # expects an env of type VaryMassRolloutWrapper (defined in rllab by me)

    # check extreme values of mass
    for m in [env.env.m0, env.env.mf]:
        env.env.set_param('m', m)
        SATus = check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps, render=False, verbose=False)
        if SATus == "REAL":
            # run again to print things
            check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps, render=True, verbose=True)
            return SATus
    i = 0
    while i < nsims and SATus == "OVERAPPROX":
        env.set_mass_randomly()
        SATus = check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps, render=False, verbose=False)
        i += 1
    # run again to print things
    check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps, render=True, verbose=True)
    return SATus

# call Marabou solver
def solve_with_marabou(network, marabou_log_dir):
    print("Solving with Marabou...")
    vals, stats, exit_code = network.solve(marabou_log_dir)
    if len(vals)>0:
        print("Input vals: ", [vals[iv] for iv in np.array(network.inputVars).flatten()])
        print("Output vals: ", [vals[iv] for iv in np.array(network.outputVars).flatten()])
    print("stats: ", stats)
    return vals, stats, exit_code

def load_network_wrapper(fprefix, fnumber, fname):
    frozen_graph = os.path.join(fprefix, fname+fnumber+".pb")
    meta_data = os.path.join(fprefix, "meta_data_"+fnumber+".txt")
    output_op_name, inputs, outputs = read_inout_metadata(meta_data)
    network = Marabou.read_tf(frozen_graph, outputName=output_op_name)
    return frozen_graph, network, inputs

def set_up_logging(fnumber):
    run_number = str(int(np.ceil(np.random.rand()*10000)))
    # make path in which to store outputs
    network_dir = '/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/MarabouLogs/network_'+fnumber
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)
    # set up marabou logging
    marabou_log_dir = os.path.join(network_dir, 'run_'+run_number+'_marabou.log')
    print(marabou_log_dir)
    if os.path.exists(marabou_log_dir): # don't overwrite old data!!!
        raise FileExistsError
    #
    # redirect to file
    logname = os.path.join(network_dir,'run_'+run_number+'_peripheral.log')
    print("log: ", logname)
    sys.stdout = open(logname, 'w')
    return logname, marabou_log_dir, network_dir, run_number
 
def check_SAT(frozen_graph, vals, bounds, nsteps):
    env = SimpleWrapper(gym.envs.make('MyPendulum-v0')) # , recordVideo=True)
    sess = load_network(frozen_graph)
    SATus = check_SAT_REAL_or_OVERAPPROX(sess, vals, env, bounds, nsteps)
    print("SATus: ", SATus)
    return SATus



    




        



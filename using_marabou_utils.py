# using marabou utils
import colored_traceback.always
import numpy as np
import tensorflow as tf
from maraboupy.MarabouUtils import *
import gym


def read_inout_metadata(meta_data):
    with open(meta_data) as f:
        line = f.readline()
        output_op_name = f.readline()[0:-1] # should contain name of output op for ffnetwork
        line = f.readline()
        inputs = f.readline().split(',')[0:-1] # discard newline character at end of line
        line = f.readline()
        outputs = f.readline()
        return output_op_name, inputs, outputs

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
    vals, stats = network.solve()

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
        print("theta for getting action: ", state_t[0]*180/np.pi, " deg")
        action = self.sess.run([self.controller_output],
            feed_dict = {
            get_output("initial_values/theta_0", graph=self.sess.graph): 
                np.array([[state_t[0]]]),
            get_output("initial_values/theta_dot_0", graph=self.sess.graph): 
            np.array([[state_t[1]]])
            })
        return action[0], {"":""}

def stepEnv(env, policy, nsteps, vals):
    # step through 2 steps of simulation
    print("begin sim.")
    env.env.reset()
    env.env.state = np.array([vals[0], vals[1]]) # set theta0 and thetadot0
    env.env.render()
    obsList = []
    for i in range(nsteps):
        action, _ = policy.get_action(env.env.state)
        obs, _, _, _ = env.env.step(action) # get theta and thetadot
        obsList.append(obs)
        env.env.render()
        input("enter to continue")
    print("end sim.")
    return obsList # thetas and thetadots

def check_SAT_REAL_or_OVERAPPROX(frozen_graph, vals, envStr, bounds):
    # check if SAT example is REAL or due to OVERAPPROX
    # gonna wanna import mypendulumenv
    # pull the controller out of the composed graph
    # SET the state of the pendulum using the sat values
    # step twice 
    # look at the state values and compare them to limits
    # then announce whether TRUE SAT or OVERAPPROX SAT
    env = gym.envs.make(envStr) #  record_video=True (how to get vidoe to record??)
    sess = load_network(frozen_graph)
    policy = Lightweight_policy(sess)
    obsList = stepEnv(env, policy, 2, vals)
    # compare values
    # compare theta values
    print("theta1 from sim: ", obsList[0][0]*180/np.pi) # theta1
    print("theta1 from SAT: ", vals[4]*180/np.pi)
    print("theta2 from sim: ", obsList[1][0]*180/np.pi) # theta1
    print("theta2 from SAT: ", vals[5]*180/np.pi)
    
    if "theta_2" in bounds.outputs_max:
        strkey = "theta_2"
    elif "thetas" in bounds.outputs_max:
        strkey = "thetas"
    a = obsList[1][0][0] <= bounds.outputs_max[strkey]
    b = obsList[1][0][0] >= bounds.outputs_min[strkey]
    if a and b:
        return "OVERAPPROX"
    else:
        return "REAL"





        



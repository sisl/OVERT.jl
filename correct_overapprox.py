# adapt simple_pendulum_multi_step_constant_dynamics.py
# make it simpler
# make it a true overapproximation, so that output set is large enough to be correct

# Features: __________________________
# multiple steps
# constant dynamics (accel) bound


import colored_traceback.always
import tensorflow as tf 
import numpy as np
import os
import joblib
from simple_overapprox_simple_pendulum import line, bound, build_sin_approx

# parsing support libs
from tensorflow.python.framework import graph_util
import parsing
from NNet.scripts.writeNNet import writeNNet


# ideas from luke: model class. state class

# constant dynamics block for pendulum
class Dynamics():
    def __init__(self, m, l): 
        with tf.name_scope("dynamics_constants"):
            self.m = m #0.25 # kg
            print("mass of pendulum: ", m)
            self.l = l #0.1 # m
            print("length of pendulum: ", l)
            self.g = 9.8
            dt = 0.05 # timestep (sec?)
            self.oomls = tf.constant([[(1/ (self.m*(self.l**2)) )]], name="torque_scaling")
            # currently accel bounds are constant but will be funcs of theta
            self.max_grav_torque = self.g/self.l
            self.deltat = tf.constant(dt*np.eye(1), dtype='float32', name="deltat")

    def accel_UB(self, theta):
        with tf.name_scope("accel_UB"):
            return tf.constant([[self.max_grav_torque]]) # prev: 50

    def accel_LB(self, theta): 
        with tf.name_scope("accel_LB"):
            return tf.constant([[-self.max_grav_torque]])

    def run(self, num, action, theta, theta_dot): 
        with tf.name_scope("Dynamics_"+str(num)):
            theta_hat = tf.add(theta, self.deltat@theta_dot, name="theta_"+str(num))
            with tf.name_scope("thdotUB"):
                theta_dot_UB = tf.add(
                            theta_dot,
                            self.deltat@(self.oomls@action + self.accel_UB(theta)),
                            name="theta_dot_UB_"+str(num)
                        )
            with tf.name_scope("thdotLB"):
                theta_dot_LB = tf.add(
                            theta_dot,
                            self.deltat@(self.oomls@action + self.accel_LB(theta)),
                            name="theta_dot_LB_"+str(num)
                        )
        return theta_hat, theta_dot_LB, theta_dot_UB

# piecewise dynamics block for pendulum
class PiecewiseDynamics():
    def __init__(self, m, l):
        self.m = m
        print("mass of pendulum: ", m)
        self.l = l
        print("length of pendulum: ", l)
        self.g = 9.8
        dt = 0.05 # timestep
        with tf.name_scope("dynamics_constants"):
            self.deltat = tf.constant(dt*np.eye(1), dtype='float32', name="deltat")
            self.oomls = tf.constant([[(1/ (self.m*(self.l**2)) )]], name="torque_scaling")
            # build_sin_approx(fun, c1, convex_reg, concave_reg)
            c1 = self.g/self.l
            print("c1: ", c1)
            fun = lambda x: c1*np.sin(x)
            with tf.name_scope("sin_approx"):
                LB, UB = build_sin_approx(fun, c1, [-np.pi, 0.0], [0.0, np.pi])
            self.accel_UB = UB
            self.accel_LB = LB
    def run(self, num, action, theta, theta_dot):
        with tf.name_scope("Dynamics_"+str(num)):
            theta_hat = tf.add(theta, self.deltat@theta_dot, name="theta_"+str(num))
            with tf.name_scope("thdotUB"):
                theta_dot_UB = tf.add(
                                theta_dot,
                                self.deltat@(self.oomls@action + self.accel_UB.tf_apply(theta)),
                                name="theta_dot_UB_"+str(num)
                            )
            # NOTE: there is elementwise multiplication in this bound: in the line() class and in the negation for creating min from tf.max
            with tf.name_scope("thdotLB"):
                theta_dot_LB = tf.add(
                                theta_dot,
                                self.deltat@(self.oomls@action + self.accel_LB.tf_apply(theta)),
                                name="theta_dot_LB_"+str(num)
                            )
        return theta_hat, theta_dot_LB, theta_dot_UB
            


class Controller():

    def __init__(self, sess):
        with tf.name_scope('Controller'):
            self.W0 = tf.Variable(np.random.rand(4,2)*.2-.1, dtype='float32')
            self.b0 = tf.Variable(np.random.rand(4,1)*.2-.1, dtype='float32')
            self.W1 = tf.constant(np.random.rand(4,4)*.2-.1, dtype='float32')
            self.b1 = tf.constant(np.random.rand(4,1)*.2-.1, dtype='float32')
            self.Wf = tf.constant(np.random.rand(1,4)*.2-.1, dtype='float32')
            self.bf = tf.constant(np.random.rand(1,1)*.2-.1, dtype='float32')
            sess.run([tf.global_variables_initializer()])
    def run(self, state0):
        state = tf.nn.relu(self.W0@state0 + self.b0)
        state = tf.nn.relu(self.W1@state + self.b1)
        statef = self.Wf@state + self.bf
        return statef

class RllabController():

    def __init__(self, policy_file, sess): #, output_node_name):
        self.sess = sess
        #self.output_name = output_node_name
        # e.g.
        # policy_file = "/Users/Chelsea/" # mac
        # policy_file = policy_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/ITWORKSrelutanh_small_network_ppo/params.pkl"
        # load policy object:
        with sess.as_default():
            data = joblib.load(policy_file)
            self.policy = data["policy"]
            print("loaded controller")

    def run(self, state0):
        # two options: either use policy.dist_info_sym whatever
        # OR "recreate" the layer using the weights, e.g.:
        # with scope("DenseLayer_0"):
        #   # weights = tf.get_tensor_by_name("W")
        #   # bias = tf.get_tensor_by_name("b")
        # etc and then multiply. can also add in different activations OR
        # transpose the weights!!!
        # try dist info sym first
        # 
        # if Wx convention, transpose
        if state0.shape[0].value > state0.shape[1].value: 
            state1 = tf.matmul(state0,
                tf.constant(np.eye(state0.shape[0].value), dtype="float32"),
                transpose_a=True)
            action = self.policy.dist_info_sym(state1, [])["mean"]
            # TODO: for multidimensional problems, action may need to be transposed back 
            return action
        else:
            return self.policy.dist_info_sym(state0, [])["mean"]

class ReluProtector():
    def __init__(self, state_dim):
        self.before = tf.constant(np.vstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
        self.after = tf.constant(np.hstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
    def apply(self, state, name=""):
        return tf.matmul(self.after, tf.nn.relu(self.before@state), name=name)

def build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, ncontroller_act):
    with tf.name_scope("assign_init_vals"):
        relu_protector = ReluProtector(1)
        theta = theta_0
        theta_dot = theta_dot_0
        theta_dot_hats = []
        for i in range(nsteps):
            theta_dot_hats.append(tf.placeholder(
                    tf.float32,
                    shape=(1,1),
                    name="theta_dot_hat_"+str(i+1)
                ))
        theta_dot_LBs = []
        theta_dot_UBs = []
        thetas = []
    #
    for i in range(nsteps):
        with tf.name_scope("get_action"):
            state = tf.constant([[1.],[0.]])@theta + tf.constant([[0.],[1.]])@theta_dot 
            action = controller.run(state)
        #
        # apply relu protection
        with tf.name_scope("relu_protection"):
            for j in range(ncontroller_act):
                theta = relu_protector.apply(theta, name="theta")
                thetas = [relu_protector.apply(t, name="theta_o") for t in thetas]
                theta_dot = relu_protector.apply(theta_dot, name="theta_dot")
                theta_dot_LBs = [relu_protector.apply(tdlb, name="tdlb") for tdlb in theta_dot_LBs]
                theta_dot_UBs = [relu_protector.apply(tdub, name="tdub") for tdub in theta_dot_UBs]
                theta_dot_hats = [relu_protector.apply(tdh, name="tdh") for tdh in theta_dot_hats]

        # input of dynamics is torque(action), output is theta theta-dot at the next timestep
        with tf.name_scope("run_dynamics"):
            theta, theta_dot_LB, theta_dot_UB = dynamics.run(num=i+1, action=action, theta=theta, theta_dot=theta_dot)
            theta_dot_LBs.append(theta_dot_LB)
            theta_dot_UBs.append(theta_dot_UB)
            thetas.append(theta) # collect all thetas from theta_1 onwards
            theta_dot = theta_dot_hats[i]
            
#
    return (thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs)

def display_ops(sess):
    g = sess.graph.as_graph_def()
    # print n for all n in graph_def.node
    print("before freezing:")
    [print(n.name) for n in g.node]
    # make a set of the ops:
    op_set = {(x.op,) for x in g.node}
    # print this set of ops
    [print(o[0]) for o in op_set]

def write_to_tensorboard(logdir, sess):
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(sess.graph) 
    train_writer.close()
    print("wrote to tensorboard log")

def write_metadata(input_list, output_list, output_op_name,directory, fid):
    filename = os.path.join(directory, "meta_data_" + fid + ".txt")
    with open(filename,'w') as file:
        file.write("concatenated output op name: \n")
        file.write("%s\n" % output_op_name)
        file.write("inputs: \n")
        for i in range(len(input_list)):
            file.write("%s," % input_list[i])
        file.write("\n")
        file.write("outputs: \n")
        [file.write("%s," % item) for item in output_list]

def write_output_metadata(output_list, output_op_name, directory, fid):
    filename = os.path.join(directory, "meta_data_" + fid + ".txt")
    with open(filename,'w') as file:
        file.write("output op name: \n")
        file.write("%s\n" % output_op_name)
        file.write("outputs: \n")
        [file.write("%s," % item) for item in output_list]

def collect_output_ops(cat_list):
    master_list = []
    for cat in cat_list:
        ops = [t.op for t in cat]
        master_list.extend(ops)
    return master_list

# assume Wx convention
# assume 1d inputs
def concat_outputs(op_list):
    n = len(op_list)
    z = np.zeros((n,1))
    #
    z[0] = 1.
    out = tf.constant(z, dtype='float32')@op_list[0].outputs[0]
    z[0] = 0.
    for i in range(1,len(op_list)):
        z[i] = 1.
        out += tf.constant(z, dtype='float32')@op_list[i].outputs[0]
        z[i] = 0.
    return out

def build_model(nsteps, 
                output_pb, 
                output_nnet, 
                tensorboard_log_dir, 
                network_dir, 
                f_id,
                policy_file=None, 
                activation_type="Relu",
                activation_fn=tf.nn.relu,
                policy_output_node_name="", 
                controller_activations=None,  
                verbose=True):

    with tf.variable_scope("initial_values"):
        theta_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_0")
        theta_dot_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_dot_0")

    # construct controller
    sess = tf.Session()
    if policy_file is None:
        print("Using random controller")
        controller = Controller(sess) # random controller
        controller_activations = 2
    else:
        print("Using controller from rllab")
        controller = RllabController(policy_file, sess)
        controller_activations = 2 ## TODO: CHECK THAT THIS IS CORRECT

    # policy works well up to here

    # construct dynamics
    """
    mass in kg (original: 0.25)
    length in meters (original: 0.1)
    """
    dynamics = Dynamics(m=0.25, l=0.1)

    # put controller and dynamics together
    thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs = build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, controller_activations)

    # condense output to only final theta
    with tf.name_scope("condense_outputs"):
        theta_out = tf.constant([[0.0]])@theta_dot + thetas[-1]

    #################### testing and writing to file from here on out ##########
    test_and_write(sess, 
                theta_0, 
                theta_dot_0, 
                nsteps, 
                thetas, 
                theta_dot, 
                theta_dot_hats, 
                theta_dot_LBs, 
                theta_dot_UBs, 
                network_dir, 
                f_id, 
                output_pb, 
                output_nnet, 
                verbose,
                activation_fn, 
                activation_type,
                tensorboard_log_dir)
    return "success"


def build_heavier_model(nsteps, 
                output_pb, 
                output_nnet, 
                tensorboard_log_dir, 
                network_dir, 
                f_id,
                policy_file=None, 
                activation_type="Relu",
                activation_fn=tf.nn.relu,
                policy_output_node_name="", 
                controller_activations=None,  
                verbose=True):

    with tf.variable_scope("initial_values"):
        theta_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_0")
        theta_dot_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_dot_0")

    # construct controller
    sess = tf.Session()
    if policy_file is None:
        print("Using random controller")
        controller = Controller(sess) # random controller
        controller_activations = 2
    else:
        print("Using controller from rllab")
        controller = RllabController(policy_file, sess)
        controller_activations = 2 ## TODO: CHECK THAT THIS IS CORRECT

    # policy works well up to here

    # construct dynamics
    """
    mass in kg (original: 0.25)
    length in meters (original: 0.1)
    """
    dynamics = Dynamics(m=0.3, l=0.1)

    # put controller and dynamics together
    thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs = build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, controller_activations)

    # condense output to only final theta
    with tf.name_scope("condense_outputs"):
        theta_out = tf.constant([[0.0]])@theta_dot + thetas[-1]

    #################### testing and writing to file from here on out ##########
    test_and_write(sess, 
                theta_0, 
                theta_dot_0, 
                nsteps, 
                thetas, 
                theta_dot, 
                theta_dot_hats, 
                theta_dot_LBs, 
                theta_dot_UBs, 
                network_dir, 
                f_id, 
                output_pb, 
                output_nnet, 
                verbose,
                activation_fn, 
                activation_type,
                tensorboard_log_dir)
    return "success"

def build_heavier_model_piecewise(nsteps, 
                output_pb, 
                output_nnet, 
                tensorboard_log_dir, 
                network_dir, 
                f_id,
                policy_file=None, 
                activation_type="Relu",
                activation_fn=tf.nn.relu,
                policy_output_node_name="", 
                controller_activations=None,  
                verbose=True):

    with tf.variable_scope("initial_values"):
        theta_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_0")
        theta_dot_0 = tf.placeholder(tf.float32, shape=(1,1), name="theta_dot_0")

    # construct controller
    sess = tf.Session()
    if policy_file is None:
        print("Using random controller")
        controller = Controller(sess) # random controller
        controller_activations = 2
    else:
        print("Using controller from rllab")
        controller = RllabController(policy_file, sess)
        controller_activations = 2 ## TODO: CHECK THAT THIS IS CORRECT

    # policy works well up to here

    # construct dynamics
    """
    mass in kg (original: 0.25)
    length in meters (original: 0.1)
    """
    dynamics = PiecewiseDynamics(m=0.3, l=0.1)

    # put controller and dynamics together
    thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs = build_multi_step_network(theta_0, theta_dot_0, controller, dynamics, nsteps, controller_activations)

    # condense output to only final theta
    with tf.name_scope("condense_outputs"):
        theta_out = tf.constant([[0.0]])@theta_dot + thetas[-1]

    #################### testing and writing to file from here on out ##########
    test_and_write(sess, 
                theta_0, 
                theta_dot_0, 
                nsteps, 
                thetas, 
                theta_dot, 
                theta_dot_hats, 
                theta_dot_LBs, 
                theta_dot_UBs, 
                network_dir, 
                f_id, 
                output_pb, 
                output_nnet, 
                verbose,
                activation_fn, 
                activation_type,
                tensorboard_log_dir)
    return "success"

def test_and_write(sess, 
                theta_0, 
                theta_dot_0, 
                nsteps, 
                thetas, 
                theta_dot, 
                theta_dot_hats, 
                theta_dot_LBs, 
                theta_dot_UBs, 
                network_dir, 
                f_id, 
                output_pb, 
                output_nnet, 
                verbose,
                activation_fn, 
                activation_type,
                tensorboard_log_dir):
    # run to make sure network is valid
    init_theta = 2*np.random.rand(1,1)-1
    init_theta_dot = 2*np.random.rand(1,1)-1
    init_theta_dot_hats = []
    for i in range(nsteps):
        init_theta_dot_hats.append(2*np.random.rand(1,1)-1)
    if verbose:
        display_ops(sess)
    # construct inputs
    feed_dict = {theta_0.name: init_theta,
                theta_dot_0.name: init_theta_dot
    }
    for i in range(nsteps):
        key = 'assign_init_vals/theta_dot_hat_'+str(i+1)+':0'
        feed_dict[key] = init_theta_dot_hats[i]
    # run with test inputs
    thetas_v, theta_dot_v, theta_dot_hats_v, theta_dot_LBs_v, theta_dot_UBs_v = sess.run([thetas, theta_dot, theta_dot_hats, theta_dot_LBs, theta_dot_UBs], feed_dict=feed_dict)
    if verbose:
        print("thetas: ", thetas_v)
        print("theta_dot: ", theta_dot_v)
        print("theta_dot_hats: ", theta_dot_hats_v)
        print("theta_dot_LBs: ", theta_dot_LBs_v)
        print("theta_dot_UBs: ", theta_dot_UBs_v)
        print("thetas")
        print([t.op.name for t in thetas])
        print("theta_dot_hats")
        print([t.op.name for t in theta_dot_hats])
        print("theta_dot_LBs")
        print([t.op.name for t in theta_dot_LBs])
        print("theta_dot_UBs")
        print([t.op.name for t in theta_dot_UBs])

    # collect output ops for parsing
    output_ops = collect_output_ops([thetas, theta_dot_hats, theta_dot_LBs, theta_dot_UBs])
    if verbose:
        print("output_ops: ", output_ops)

    # outputting a frozen .pb file ~ ~ ~
    # if parsing with maraboupy parser, can only have a single output node
    if output_pb:
        write_pb(sess, output_ops, feed_dict, network_dir, f_id, verbose)
    write_to_tensorboard(tensorboard_log_dir, sess)
    # if want to output a .nnet, must convert to being ff
    if output_nnet:
        ffnet = write_nnet(sess, output_ops, feed_dict, activation_fn, activation_type, thetas_v, theta_dot_hats_v, theta_dot_LBs_v, theta_dot_UBs_v, network_dir, nsteps, f_id, verbose)

    # write to tensorboard
    write_to_tensorboard(tensorboard_log_dir, sess)
    # next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/new_approach_3605 --host=localhost --port=1234


def write_pb(sess, output_ops, feed_dict, network_dir, f_id, verbose):
    with tf.name_scope("Output"):
        output = concat_outputs(output_ops)
    # convert variables to constants
    output_graph_def = graph_util.convert_variables_to_constants(
    sess, # sess used to retrieve weights
    sess.graph.as_graph_def(), # graph def used to retrieve nodes
    [output.op.name] # output node names used to select useful nodes
    )
    if verbose:
        display_graph_def_info(output_graph_def)
    # write to file!
    output_graph_name = os.path.join(network_dir, "graph_def_"+f_id+".pb")
    with tf.gfile.GFile(output_graph_name, "w") as f:
        f.write(output_graph_def.SerializeToString())
    # collect output op names for ops pre-condensation
    output_op_names = [op.name for op in output_ops]
    # collect input list to write to metadata
    input_list = [k[:-2] for k in feed_dict.keys()]
    # write metadata to file (used in loading and verifying controller)
    write_metadata(input_list, output_op_names, output.op.name, network_dir, f_id)


def write_nnet(sess, output_ops, feed_dict, activation_fn, activation_type, thetas_v, theta_dot_hats_v, theta_dot_LBs_v, theta_dot_UBs_v, network_dir, nsteps, f_id, verbose):
    output_op_names = [op.name for op in output_ops]
    output_graph_def = graph_util.convert_variables_to_constants(
    sess, # sess used to retrieve weights
    sess.graph.as_graph_def(), # graph def used to retrieve nodes
    output_op_names # output node names used to select useful nodes
    )
    if verbose:
        display_graph_def_info(output_graph_def)
    # load graph with constants back into TF
    tf.import_graph_def(output_graph_def)
    g = tf.get_default_graph()
    # get output ops
    imported_output_ops = [g.get_operation_by_name("import/"+node) for node in output_op_names]
    ##################################################################################
    # Parse into feed-forward network
    W,b, input_list = parsing.parse_network_wrapper(imported_output_ops, activation_type, sess)
    ##################################################################################
    #
    #
    # test: ~ ~ ~ 
    # turn into its own tf network to check equivalency
    W.reverse()
    b.reverse()
    n_inputs = len(feed_dict) # assumes outputs are 1D
    ff_input = tf.placeholder(tf.float32, shape=(n_inputs,1), name="state0")
    ffnet = parsing.create_tf_network(W,b,inputs=ff_input, activation=activation_fn, act_type=activation_type, output_activated=False)
    # create feed dict for ffnetwork
    ff_feed_dict = {
            ff_input: np.vstack([feed_dict[i[7:]+":0"] for i in input_list]) # strip off the "import" part of every input in input_list
        }
    ff_output_tensors_v = sess.run(ffnet, feed_dict=ff_feed_dict)
    if verbose:
        print("condensed output: ", ff_output_tensors_v)
    # check equivalency
    assert(all(abs(np.array(thetas_v).flatten() - ff_output_tensors_v[0:nsteps].flatten()))<1e-4) # thetas
    assert(all(abs(np.array(theta_dot_hats_v).flatten() - ff_output_tensors_v[nsteps:2*nsteps].flatten()))<1e-4) # theta dot hats
    assert(all(abs(np.array(theta_dot_LBs_v).flatten() - ff_output_tensors_v[2*nsteps:3*nsteps].flatten())<1e-4)) # theta dot LBs
    assert(all(abs(np.array(theta_dot_UBs_v).flatten() - ff_output_tensors_v[3*nsteps:].flatten())<1e-4))# theta dot UBs
    #
    #
    # Write to .nnet file
    # and write accompanying file with meta data like order of inputs and outputs
    num_inputs = 2+nsteps
    means = [0.]*(num_inputs+1)
    ranges = [1.]*(num_inputs+1)
    inMins = [0.]*num_inputs
    inMaxs = [0.]*num_inputs
    fileName = os.path.join(network_dir, "overrapprox_const_dyn_"+str(nsteps)+"_steps_"+str(f_id)+".nnet")
    writeNNet(W,b,inputMins=inMins,inputMaxes=inMaxs,means=means,ranges=ranges, order='Wx', fileName=fileName)
    imported_output_op_names = [o.name for o in imported_output_ops]
    write_metadata(input_list, imported_output_op_names, ffnet.op.name, network_dir, f_id)
    return ffnet

def display_graph_def_info(output_graph_def):
    # print op list to make sure its only stuff we can handle
    print("final op set: ", {(x.op,) for x in output_graph_def.node})
    print("%d ops in the final graph." % len(output_graph_def.node))






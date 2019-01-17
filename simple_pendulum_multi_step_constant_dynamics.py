# simple pendulum problem

# theta_dd = F/ml  - gsin(theta)/l
# theta_dd = Tau/(ml^2) - (g/l)*sin(theta)
import colored_traceback.always
import tensorflow as tf 
import numpy as np
from simple_overapprox_simple_pendulum import line, bound, build_sin_approx
import os
import joblib
from rllab.sampler.utils import rollout
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import sandbox.rocky.tf.core.layers as L
from tensorflow.python.framework import graph_util
import parsing
from NNet.scripts.writeNNet import writeNNet

output_flag = "one network"
verbose = True
f_id = str(int(np.round(np.random.rand()*5000)))

# inputs: state and action yields next state
class Dynamics():
    def __init__(self, bound_flag): 
        with tf.name_scope("dynamics_constants"):
            self.m = 0.25 # kg
            self.l = 0.1 # m
            self.oomls = tf.constant([[(1/ (self.m*(self.l**2)) )]], name="torque_scaling")
            if bound_flag == "UB":
                self.accel_B = tf.constant([[50.]])
            elif bound_flag == "LB":
                self.accel_B = tf.constant([[-50.]])
            else:
                raise NotImplementedError
            self.deltat = tf.constant(0.05*np.eye(2), dtype='float32', name="delta_t")
    
    def run(self, num, action, state): 
        with tf.name_scope("Dynamics_"+str(num)):
            theta_d = tf.constant([[0.,1.],[0.,0.]])@state
            theta_dd = tf.constant([[0.],[1.]])@(self.oomls@action + self.accel_B)
            change = self.deltat@theta_d + self.deltat@theta_dd
            state_tp1 = state + change
            return state_tp1

sess = tf.Session()
print("initialized session")
# Initialize theta and theta-dot
with tf.variable_scope("initial_values"):
    state_UB_0 = tf.placeholder(tf.float32, shape=(2,1), name="state_UB")
    # theta, theta dot, UB
    state_LB_0 = tf.placeholder(tf.float32, shape=(2,1), name="state_LB")

###########################################################
# load controller! :)
###########################################################
# policy_file = "/Users/Chelsea/" # mac
# policy_file = policy_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/relu_small_network_vpg_capped_action_trying_simpler_dense_layer/params.pkl"

# # load policy object:
# with sess.as_default():
#   #with tf.name_scope("Controller"):
#   data = joblib.load(policy_file)
#   policy = data["policy"]
#   print("loaded controller")
#   g = sess.graph.as_graph_def()
#   #[print(n.name) for n in g.node]
#   output_node_name = "policy/mean_network/output"
#   output_graph_def = graph_util.convert_variables_to_constants(
#       sess, # sess used to retrieve weights
#       g, # graph def used to retrieve nodes
#       output_node_name.split(",") # output node names used to select useful nodes
#       )
#   print("op set: ", {(x.op,) for x in output_graph_def.node})

#####################################
# dummy controller
#####################################
class Controller():

    def __init__(self):
        with tf.name_scope('Controller'):
            self.W0 = tf.Variable(np.random.rand(4,2)*2-1, dtype='float32')
            self.b0 = tf.Variable(np.random.rand(4,1)*2-1, dtype='float32')
            self.W1 = tf.constant(np.random.rand(4,4)*2-1, dtype='float32')
            self.b1 = tf.constant(np.random.rand(4,1)*2-1, dtype='float32')
            self.Wf = tf.constant(np.random.rand(1,4)*2-1, dtype='float32')
            self.bf = tf.constant(np.random.rand(1,1)*2-1, dtype='float32')

    def run(self, state0):
        state = tf.nn.relu(self.W0@state0 + self.b0)
        state = tf.nn.relu(self.W1@state + self.b1)
        statef = self.Wf@state + self.bf
        return statef

class ReluProtector():
    def __init__(self, state_dim):
        self.before = tf.constant(np.vstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
        self.after = tf.constant(np.hstack([np.eye(state_dim), -np.eye(state_dim)]), dtype='float32')
    def apply(self, state):
        return self.after@(tf.nn.relu(self.before@state))

# Controller -> Dynamics loop
def build_multi_step_network(state_UB_0, state_LB_0, controller, ncontroller_act, dynamics_UB, dynamics_LB, nsteps):
    with tf.name_scope("assign_init_vals"):
        relu_protector = ReluProtector(state_UB_0.shape[0].value)
        state_UB = state_UB_0
        state_LB = state_LB_0
    for i in range(nsteps):
        with tf.name_scope("get_actions"):
            #action_UB, = L.get_output([policy._l_mean], tf.transpose(state_UB))
            #action_LB, = L.get_output([policy._l_mean], tf.transpose(state_LB))
            action_UB = controller.run(state_UB)
            action_LB = controller.run(state_LB)
        #
        # apply relu protection
        with tf.name_scope("relu_protection"):
            for i in range(ncontroller_act):
                state_UB = relu_protector.apply(state_UB)
                state_LB = relu_protector.apply(state_LB)

        # input of dynamics is torque(action), output is theta theta-dot at the next timestep
        with tf.name_scope("run_dynamics"):
            state_UB = dynamics_UB.run(num=i, action=action_UB, state=state_UB)
            state_LB = dynamics_LB.run(num=i, action=action_LB, state=state_LB)
    #
    return (state_UB, state_LB)

# build controller->dynamics loop
controller = Controller()
ncontroller_act = 2
nsteps = 3
dynamics_UB = Dynamics(bound_flag="UB")
dynamics_LB = Dynamics(bound_flag="LB")
state_UB, state_LB = build_multi_step_network(state_UB_0, state_LB_0, controller, ncontroller_act, dynamics_UB, dynamics_LB, nsteps)
with tf.name_scope("Output_UB"):
    state_UB_final = tf.constant([[1.,0.]])@state_UB
with tf.name_scope("Output_LB"):
    state_LB_final = tf.constant([[1.,0.]])@state_LB

init_UB = np.array([[1.0], [2.0]])
init_LB = (1/2)*init_UB
feed_dict_UB = {
    state_UB_0: init_UB,
}
feed_dict_LB = {
    state_LB_0: init_LB,
}
sess.run(tf.global_variables_initializer())
state_UB_final_out, = sess.run([state_UB_final], feed_dict=feed_dict_UB)
state_LB_final_out, = sess.run([state_LB_final], feed_dict=feed_dict_LB)
print("theta_UB: ", state_UB_final_out)
print("theta_LB: ", state_LB_final_out)



def get_state(n_activations):
    state = tf.placeholder(shape=(1,1), dtype='float32')
    relu_protector = ReluProtector(1)
    for i in range(n_activations):
        state = relu_protector.apply(state)
    return state

if output_flag == "two networks":
    # get current graph
    g = sess.graph.as_graph_def()
    if verbose:
        # see how many unique ops in the graph before converting vars to consts
        # print n for all n in graph_def.node
        print("before freezing:")
        [print(n.name) for n in g.node]
        # make a set of the ops:
        op_set = {(x.op,) for x in g.node}
        # print this set of ops
        [print(o[0]) for o in op_set]
    # two networks
    # UB then LB
    output_node_names = "Output_UB/matmul,Output_LB/matmul"
    output_graph_def = graph_util.convert_variables_to_constants(
    sess, # sess used to retrieve weights
    g, # graph def used to retrieve nodes
    output_node_names.split(",") # output node names used to select useful nodes
    )
    graph_defs = [output_graph_def]
    output_nodes = [[o] for o in output_node_names.split(",")]
    if verbose:
        # ops in final graph
        print("op set: ", {(x.op,) for x in output_graph_def.node})
elif output_flag == "one network":
    # one network
    with tf.name_scope("state"):
        state = get_state(ncontroller_act*nsteps)
    # concat three outputs: UB theta, theta, LB theta
    with tf.name_scope("concat_output"):
        concat_state = tf.constant([[1.0],[0.],[0.]])@state_UB_final + tf.constant([[0.0],[1.],[0.]])@state + tf.constant([[0.0],[0.],[1.]])@state_LB_final
    # get current graph
    g = sess.graph.as_graph_def()
    if verbose:
        # see how many unique ops in the graph before converting vars to consts
        # print n for all n in graph_def.node
        print("before freezing:")
        [print(n.name) for n in g.node]
        # make a set of the ops:
        op_set = {(x.op,) for x in g.node}
        # print this set of ops
        [print(o[0]) for o in op_set]
    output_node_name = "concat_output/add_1"
    output_graph_def = graph_util.convert_variables_to_constants(
    sess, # sess used to retrieve weights
    g, # graph def used to retrieve nodes
    output_node_name.split(",") # output node names used to select useful nodes
    )
    graph_defs = [output_graph_def]
    output_nodes = [[output_node_name]]
    if verbose:
        # ops in final graph
        print("op set: ", {(x.op,) for x in output_graph_def.node})
else:
    raise NotImplementedError


# output_node_name = "UB_LB_concat/add"
# output_graph_def = graph_util.convert_variables_to_constants(
#     sess, # sess used to retrieve weights
#     g, # graph def used to retrieve nodes
#     output_node_name.split(",") # output node names used to select useful nodes
#     )
# #print("all nodes: ", [(x.op, x.name) for x in output_graph_def.node])
# print("op set: ", {(x.op,) for x in output_graph_def.node})

# clear graphs and sessions and import only the parsed graphs
sess.close()
tf.reset_default_graph()
s2 = tf.Session()
with s2.as_default():
    [tf.import_graph_def(g) for g in graph_defs]
    g = tf.get_default_graph()

# parse!!!! to list of weights and biases
W_list = []
b_list = []
for i in range(len(output_nodes)):
    output_ops = [g.get_operation_by_name("import/"+node) for node in output_nodes[i]]
    print("got operation(s)")
    W,b = parsing.parse_network(output_ops, [], [], [], [], 'Relu', s2)
    print("parsed!")
    # create ff network and ensure it produces the same result as the original
    W.reverse()
    b.reverse()
    if output_flag == "one network":
        with tf.name_scope("ff_net"):
            state_net = tf.placeholder(tf.float32, shape=(5,1), name="state0")
            net = parsing.create_tf_network(W,b,inputs=state_net, activation=tf.nn.relu, act_type='Relu', output_activated=False)
        feed_dict = {
            # UB_theta, LB_theta, theta
            state_net: np.vstack([init_UB, init_LB, np.array([[42.]])])
        }
        state_after_parsing, = s2.run([net], feed_dict=feed_dict)
        print("state  after parsing", state_after_parsing)
        assert(abs(state_UB_final_out - state_after_parsing[0])<1e-4)
        assert(abs(state_LB_final_out - state_after_parsing[2])<1e-4)
    elif output_flag == "two networks":
        W_list.append(W)
        b_list.append(b)
        if i == 0:
            aux = "UB"
        else:
            aux = "LB"
        with tf.name_scope("ff_net_"+aux):
            state_net = tf.placeholder(tf.float32, shape=(2,1), name="state0")
            net = parsing.create_tf_network(W,b,inputs=state_net, activation=tf.nn.relu, act_type='Relu', output_activated=False)
        if i == 0:
            init = init_UB
        elif i == 1:
            init = init_LB
        feed_dict = {
            state_net: init
        }
        state_after_parsing, = s2.run([net], feed_dict=feed_dict)
        print("state  after parsing", state_after_parsing)
        if i == 0:
            assert(abs(state_UB_final_out - state_after_parsing)<1e-4)
        elif i == 1:
            assert(abs(state_LB_final_out - state_after_parsing)<1e-4)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    

print("Tests pass! Networks are equivalent.")
# write original AND parsed graphs to tensorboard summary file
LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/constant_dynamics_handcrafted_policy_"+f_id
train_writer = tf.summary.FileWriter(LOGDIR) #, sess.graph)
train_writer.add_graph(tf.get_default_graph()) # TODO: add the filtered graph only! # sess.graph
train_writer.close()
print("wrote to tensorboard log")

# next run at command line, e.g.:  tensorboard --logdir=/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/UGH_multi_2

# write to .nnet file ########################
if output_flag == "one network":
    means = [0.,0.,0.,0.,0.,0.]
    ranges = [1., 1., 1., 1., 1., 1.]
    inMins = [-1.,-50., -1., -50., -100.]
    inMaxs = [1.,50., 1., 50., 100.]
    fileName = "/Users/Chelsea/Dropbox/AAHAA/src/nnet_files/const_dyn_one_network_"+f_id
    writeNNet(W,b,inputMins=inMins,inputMaxes=inMaxs,means=means,ranges=ranges, order='Wx', fileName=fileName)
elif output_flag == "two networks":
    means = [0.,0.,0.]
    ranges = [1., 1., 1.]
    inMins = [-1.,-50.]
    inMaxs = [1.,50.]
    fileName_UB = "/Users/Chelsea/Dropbox/AAHAA/src/nnet_files/const_dyn_UB_"+f_id
    writeNNet(W_list[0],b_list[0],inputMins=inMins,inputMaxes=inMaxs,means=means,ranges=ranges, order='Wx', fileName=fileName_UB)
    fileName_LB = "/Users/Chelsea/Dropbox/AAHAA/src/nnet_files/const_dyn_LB_"+f_id
    writeNNet(W_list[1],b_list[1],inputMins=inMins,inputMaxes=inMaxs,means=means,ranges=ranges, order='Wx', fileName=fileName_LB)

##########################################################







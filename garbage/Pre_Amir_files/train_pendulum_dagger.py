import colored_traceback.always
from qj_global import qj

from gym.envs.registration import register

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.core.network import MLP

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, PerlmutterHvp
from rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from rllab.sampler.utils import rollout

from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.policies.simple_RNN_policy_categorical import CategoricalSimpleRNNPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.ppo import PPO
from sandbox.rocky.tf.algos.npo import NPO
from sandbox.rocky.tf.algos.dagger import Dagger
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

import tensorflow as tf
import os
import numpy as np
import joblib
import rllab.misc.logger as logger
from tensorflow.python.framework import graph_util
from export_graph import export_graph
from using_tensorboard import import_to_tb

from ensemble_dagger_code_aaai_supp.dagger.decision_rules import vanilla_dagger_decision_rule

from OverApprox.setup_logging import setup_logging
from OverApprox.relu_approximations import relu_tanh


#log_dir = "/home/csidrane/" # linux
log_dir = "/Users/Chelsea/" # mac
log_dir = log_dir + "Dropbox/AAHAA/src/rllab/data/local/dagger/"
log_dir = log_dir + "actual_actions"
if os.path.exists(log_dir): # don't overwrite old data!!!
    raise FileExistsError
print("logging to :", log_dir)

env = TfEnv(GymEnv("MyPendulum-v0", record_video=False))
print("Created environment")

novice_sess = tf.Session() #tf.InteractiveSession()
expert_sess = tf.Session()
print("created tf sessions")
print("novice session object: ", novice_sess)

with novice_sess.as_default():
    novice = GaussianMLPPolicy(
        name="novice",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64, 32, 32),
        hidden_nonlinearity=tf.nn.tanh #relu_tanh,
        #output_nonlinearity=relu_tanh
        # idea: define new tf nonlinearity that is a cap, made up of two relus
    )
    print("Created novice policy of type: ", type(novice))

#expert_file = "/home/csidrane/" # linux
expert_file = "/Users/Chelsea/" # mac

expert_file = expert_file + "Dropbox/AAHAA/src/rllab/data/local/experiment/EXPERT_tanh_inside_inf_torque_unormalized_env/params.pkl"

test_expert = False
with expert_sess.as_default():
    data = joblib.load(expert_file)
    expert = data["policy"]
    # can sim policy to test it! :DDD
    if test_expert:
        path = rollout(env, expert, max_path_length=500, animated=True, speedup=2, always_return_paths=True)
        print("reward: ", sum(path["rewards"]))
        input("enter to continue")

beta_decay= 0.98
print("beta decay: ", beta_decay)

DR = vanilla_dagger_decision_rule(novice, expert, beta_decay, novice_sess=novice_sess, expert_sess=expert_sess)

# init DAgger!
algo = Dagger(
    env=env,
    policy=novice,
    expert=expert,
    decision_rule=DR,
    numtrajs=20, 
    n_itr=5,
    max_path_length = 500,
    plot=True,
    pause_for_plot=False,
    log_file=log_dir
)

print("Created algorithm object of type: ", type(algo))

setup_logging(log_dir, algo, env, novice, expert, DR)

# train!
print("Training...")
novice_sess.run(tf.global_variables_initializer())
# a check to make sure that the global variable initialization hasn't overwritten the expert policy
# with expert_sess.as_default():
#     # can sim policy to test it! :DDD
#     path = rollout(env, expert, max_path_length=200, animated=True, speedup=2, always_return_paths=True)
#     print("reward: ", sum(path["rewards"]))
#     input("enter to continue")
algo.train()
print("Done training.")
expert_sess.close()

# # export graph
# print("Exporting graph...")
# with novice_sess.as_default():
#     input_graph_def = tf.get_default_graph().as_graph_def()
# output_node = "policy/prob_network/output" # for MLP
# #output_node = "novice/prob_network/cell_output"
# export_graph(log_dir, output_node, novice, novice_sess, input_graph_def, examine=False)

# novice_sess.close()

# # # open in tensorboard
# import_to_tb(log_dir)






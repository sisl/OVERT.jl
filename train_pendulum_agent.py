import colored_traceback.always

from sandbox.rocky.tf.algos.ppo import PPO
from sandbox.rocky.tf.algos.vpg import VPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline, LinearFeatureBaselineTranspose
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.envs.vary_wrapper import VaryWrapper
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import SimpleGaussianMLPPolicy, GaussianMLPPolicy, GaussianMLP2Policy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.policies.simple_RNN_policy_categorical import CategoricalSimpleRNNPolicy
import tensorflow as tf
tf.enable_eager_execution()
from OverApprox.relu_approximations import relu_tanh, linearized_tanh


#########  A SCRIPT FOR TRAINING AN EXPERT WITH TANH ACTIVATION

def run_task(*_):
    
    env = TfEnv(GymEnv("MyPendulum-v0", record_video=False))
    #
    policy = GaussianGRUPolicy( #GaussianMLP2Policy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        #hidden_sizes= (4,4), #(128, 128, 128, 128, 128, 128),
        hidden_dim=4,
        #layer_dim=4,
        hidden_nonlinearity=tf.nn.relu, #linearized_tanh
        # tf.nn.relu, # relu_tanh
        #output_nonlinearity=tf.nn.sigmoid
        # idea: define new tf nonlinearity that is a cap, made up of two relus
    )
    #
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    #
    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=1, #000,
        discount=0.99,
        step_size=0.0075, # 0.01
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=5,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    exp_name="relu_small_network_ppo_capped_action_simpler_dense_layer_xW_learn_std_smaller_learning_rate",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=0,
    # plot=True,
)

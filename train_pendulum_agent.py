import colored_traceback.always
from qj_global import qj

from sandbox.rocky.tf.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import tensorflow as tf
from OverApprox.relu_approximations import relu_tanh


#########  A SCRIPT FOR TRAINING AN EXPERT WITH TANH ACTIVATION

def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = TfEnv(GymEnv("MyPendulum-v0", record_video=False))
    #
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(4,4), #(128, 128, 128, 128, 128, 128),
        hidden_nonlinearity=relu_tanh, #tf.nn.relu,
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
        n_itr=200,
        discount=0.99,
        step_size=0.01, # 0.01
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
    exp_name="relutanh_small_network_ppo",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)

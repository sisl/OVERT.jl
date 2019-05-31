import colored_traceback.always

from sandbox.rocky.tf.algos.ppo import PPO
from sandbox.rocky.tf.algos.vpg import VPG
import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline, LinearFeatureBaselineTranspose
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.envs.vary_wrapper import VaryMassEnv
from sandbox.rocky.tf.samplers.vectorized_varying_sampler import VectorizedVaryingSampler
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import SimpleGaussianMLPPolicy, GaussianMLPPolicy, GaussianMLP2Policy
import tensorflow as tf
tf.enable_eager_execution()
from OverApprox.relu_approximations import relu_tanh, linearized_tanh


def run_task(*_):
    
    n_itr = 1000
    env = VaryMassEnv(GymEnv("MyPendulum-v0", record_video=False),
                  m0=0.2, 
                  mf=0.3,
                  iters=n_itr)
    #
    policy = GaussianMLP2Policy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes= (4,4), #(128, 128, 128, 128, 128, 128),
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
        n_itr=n_itr,
        discount=0.99,
        step_size=0.0075, # 0.01
        sampler_cls=VectorizedVaryingSampler
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
    exp_name="curriculum_training_"+str(int(np.ceil(np.random.rand()*50000))),
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=0,
    # plot=True,
)

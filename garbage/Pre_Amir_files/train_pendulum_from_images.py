import colored_traceback.always

from sandbox.rocky.tf.algos.ppo import PPO
from sandbox.rocky.tf.algos.vpg import VPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline, LinearFeatureBaselineTranspose
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.policies.gaussian_mlp_policy import SimpleGaussianMLPPolicy, GaussianMLPPolicy, GaussianMLP2Policy
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
import tensorflow as tf
tf.enable_eager_execution()
from OverApprox.relu_approximations import relu_tanh, linearized_tanh


def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = TfEnv(GymEnv("MyPendulum-v1", record_video=False))
    #
    policy = GaussianConvPolicy(
        name="policy",
        env_spec=env.spec,
        conv_filters = [3], # how many conv layers. e.g. this is one layer with 3 fitlers (I think)
        conv_filter_sizes = [5, 5, 5],
        conv_strides = [3, 3, 3],
        conv_pads = ['SAME', 'SAME', 'SAME'],
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes= (16,4), #(128, 128, 128, 128, 128, 128),
        hidden_nonlinearity=tf.nn.relu, #linearized_tanh
        output_nonlinearity=None,
    )
    #
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    #
    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5, #4000,
        max_path_length=env.horizon,
        n_itr=2, #1000,
        discount=0.99,
        step_size=0.0075, # 0.01
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

# profiling code
# import cProfile, pstats, io
# pr = cProfile.Profile()
# pr.enable()
run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=5,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    exp_name="convnet_ppo_"+str(np.ceil(np.random.rand()*1000)),
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=0,
    # plot=True,
)
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
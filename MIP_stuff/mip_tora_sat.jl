include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
include("example_dynamics.jl")
include("satisfiability.jl")
include("monte_carlo_simulation.jl")

using LazySets
using Dates

dt = 0.1
N_OVERT = 10
n_timesteps = 10
input_set_0 = Hyperrectangle(low = [1., 1., 1., 1.], high = [1.2, 1.2, 1.2, 1.2])
unsafe_set  = Hyperrectangle(low = [1.5, -1., 2., 0.], high = [2.5, 0., 3., 1.])
input_vars = [:x1, :x2, :x3, :x4]
control_vars = [:c1]
dynamics = tora_dynamics_overt
update_rule = tora_update_rule
exact_dynamics = tora_dynamics
#network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_simple_tora.nnet"
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_complex_tora.nnet"
last_layer_activation = Id()

all_sets, all_counter_examples, violation  = satisfiability_search(dynamics, update_rule, network_file, input_set_0,
            unsafe_set, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

plot_satisfiability(unsafe_set, all_sets, all_counter_examples, network_file, exact_dynamics, violation)

include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
include("example_dynamics.jl")
include("monte_carlo_simulation.jl")
include("satisfiability.jl")

using LazySets
using Dates


dt = 0.1
N_overt = -1
n_timesteps = 40
input_set_0 = Hyperrectangle(low = [1., 1.], high = [2., 2.])
input_vars = [:th, :dth]
control_vars = [:T]
dynamics = single_pend_dynamics_overt
timed_dynamics = single_pend_dynamics_overt_timed
true_dynamics = single_pend_dynamics
update_rule = single_pend_update_rule
network_file, last_layer_activation = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_simple_single_pend.nnet", Id()
concrete_sets, symbolic_set = symbolic_bound(n_timesteps, update_rule, dynamics, timed_dynamics, network_file,
input_set_0, input_vars, control_vars, last_layer_activation, dt, N_overt)


n_sim = 1000000
simulated_sets, simulated_x = monte_carlo_simulate(true_dynamics, network_file, last_layer_activation,
                                    input_set_0, n_sim, n_timesteps, dt)

fig = plot_output_sets(concrete_sets, linecolor=:black)
fig = plot_output_sets(concrete_sets[end:end], fig=fig, linecolor=:black, linewidth=8)
fig = plot_output_sets([symbolic_set], fig=fig, linecolor=:green)
fig = plot_output_hist(simulated_x, n_timesteps; fig=fig, nbins=150)

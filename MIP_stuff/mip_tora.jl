include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
include("example_dynamics.jl")
using LazySets

dt = 0.1
N_OVERT = 5
n_timesteps = 10
input_set_0 = Hyperrectangle(low = [1., 1., 1., 1.], high = [2., 2., 2., 2.])
input_vars = [:x1, :x2, :x3, :x4]
control_vars = [:c1]
dynamics = tora_dynamics_overt
update_rule = tora_update_rule
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_simple_tora.nnet"
last_layer_activation = Id()
out_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in out_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

n_sim = 1000000
out_sets_simulated = monte_carlo_simulate(tora_dynamics, network_file, Id(),
                                          input_set_0, n_sim, n_timesteps, dt)
fig = plot_output_sets(out_sets)
fig = plot_output_sets(out_sets_simulated, fig=fig, color=:red)

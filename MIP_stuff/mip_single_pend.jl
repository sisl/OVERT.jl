
OVERAPPROX_PATH = "/home/amaleki/Dropbox/stanford/Python/OverApprox"
include("$OVERAPPROX_PATH/OverApprox/src/overapprox_nd_relational.jl")
include("$OVERAPPROX_PATH/OverApprox/src/overt_parser.jl")
include("$OVERAPPROX_PATH/MIP_stuff/overt_to_mip.jl")
include("$OVERAPPROX_PATH/MIP_stuff/read_net.jl")

using LazySets

function single_pend_dynamics_overt(range_dict, N_overt)
    v1 = :(T + sin(th) - 0.2*dth)
    v1_oA = overapprox_nd(v1, range_dict; N=N_overt)
    return v1_oA, [v1_oA.output]
end

function single_pend_dynamics(x, u)
    dx1 = x[2]
    dx2 = u[1] + sin(x[1]) - 0.2 * x[2]
    return [dx1, dx2]
end

function single_pend_update_rule(input_vars, control_vars, overt_output_vars)
    ddth = overt_output_vars[1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

dt = 0.1
N_OVERT = 10
n_timesteps = 20
input_set_0 = Hyperrectangle(low = [1., 1.], high = [2., 2.])
input_vars = [:th, :dth]
control_vars = [:T]
dynamics = single_pend_dynamics_overt
update_rule = single_pend_update_rule
network_file, last_layer_activation = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/controller_simple_single_pend.nnet", Id()
out_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in out_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

n_sim = 10000
out_sets_simulated = monte_carlo_simulate(single_pend_dynamics, network_file, Id(),
                                    input_set_0, n_sim, n_timesteps, dt)
fig = plot_output_sets(out_sets[1:2:end])
fig = plot_output_sets(out_sets_simulated[1:2:end], fig=fig, color=:red)

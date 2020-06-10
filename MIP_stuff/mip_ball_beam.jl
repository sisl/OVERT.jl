include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
using LazySets

function ball_beam_dynamics(x, u)
    dx1 = x[2]
    dx2 = 1.6*x[3]^3 - x[3] + x[1]*x[4]^2
    dx3 = x[4]
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end

function ball_beam_dynamics_overt(range_dict, N_overt)
    v2 = :(1.6*x3^3 - x3 + x1*x4^2)
    oA_out = overapprox_nd(v2, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function ball_beam_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => control_vars[1])
    return integration_map
end

dt = 0.1
N_OVERT = 5
n_timesteps = 5
input_set_0 = Hyperrectangle(low = [1., 1., 1., 1.], high = [2., 2., 2., 2.])
input_vars = [:x1, :x2, :x3, :x4]
control_vars = [:c1]
dynamics = ball_beam_dynamics_overt
update_rule  = ball_beam_update_rule
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/controller_simple_ball_beam.nnet"
last_layer_activation = Id()
out_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in out_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

n_sim = 1000000
out_sets_simulated = monte_carlo_simulate(ball_beam_dynamics, network_file, Id(),
                                          input_set_0, n_sim, n_timesteps, dt)
fig = plot_output_sets(out_sets)
fig = plot_output_sets(out_sets_simulated, fig=fig, color=:red)

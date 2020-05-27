include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
using LazySets

function ball_beam_dynamics(range_dict, N_overt)
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
N_OVERT = 3
n_timesteps = 5
input_set_0 = Hyperrectangle(low = [-1., -1., -2., -2.], high = [1., 1., 2., 2.])
input_vars = [:x1, :x2, :x3, :x4]
control_vars = [:c1]
dynamics, update_rule = ball_beam_dynamics, ball_beam_update_rule
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/controller_simple_ball_beam.nnet"
last_layer_activation = Id()
all_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in all_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

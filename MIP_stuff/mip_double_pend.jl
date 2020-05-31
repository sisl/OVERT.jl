

include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
using LazySets

function double_pend_dynamics(x, u)
    th1, th2, u1, u2 = x
    T1, T2 = u
    dth1 = u1
    dth2 = u2
    du1 = (16*T1 - sin(2*th1 - 2*th2)*u1^2 - 2*sin(th1 - th2)*u2^2 +
           2*sin(th1 - 2*th2) + 6*sin(th1) - 16*T2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2))
    du2 = (2*sin(th1 - th2)*u1^2 + 16*T2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
            - sin(th1 - th2)*u2^2 + 8*T1))/(2 - cos(th1 - th2)^2)
    dx = [dth1, dth2, du1, du2]
    return dx
end
#
# function double_pend_dynamics_overt(range_dict, N_overt)
#     v1 = :((16*T1 - sin(2*th1 - 2*th2)*u1^2 - 2*sin(th1 - th2)*u2^2 +
#            2*sin(th1 - 2*th2) + 6*sin(th1) - 16*T2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2)))
#     v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
#     range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))
#
#     v2 = :((2*sin(th1 - th2)*u1^2 + 16*T2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
#             - sin(th1 - th2)*u2^2 + 8*T1))/(2 - cos(th1 - th2)^2))
#     v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
#     range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))
#
#     oA_out = add_overapproximate([v1_oA, v2_oA])
#     return oA_out, [v1_oA.output, v2_oA.output]
# end

function double_pend_dynamics_overt(range_dict, N_overt)
    v1 = :(sin(th1))
    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))

    v2 = :(sin(th2))
    v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))

    v3 = :(sin(th1-th2))
    v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

    v4 = :(cos(th1-th2))
    v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

    v5 = :($(v3_oA.output)*u1^2)
    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))

    v6 = :($(v3_oA.output)*u2^2)
    v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))

    v7 = :(sin(th1-2*th2))
    v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))

    v8 = :(($(v7_oA.output) - $(v6_oA.output) + 8*T1 + 3*$(v1_oA.output) -$(v4_oA.output)*(8*T2 + $(v5_oA.output)))/(2-$(v4_oA.output)^2))
    v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)

    v9 = :((2*$(v5_oA.output) + 16*T2 + 4*$(v2_oA.output) -$(v4_oA.output)*(8*T1 - $(v6_oA.output) + 4*$(v1_oA.output)))/(2-$(v4_oA.output)^2))
    v9_oA = overapprox_nd(v9, range_dict; N=N_OVERT)

    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA, v9_oA])
    return oA_out, [v6_oA.output, v7_oA.output, v8_oA.output]
end


function double_pend_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[3],
                           input_vars[2] => input_vars[4],
                           input_vars[3] => overt_output_vars[1],
                           input_vars[4] => overt_output_vars[2])
    return integration_map
end

dt = 0.1
N_OVERT = 3
n_timesteps = 10
input_set_0 = Hyperrectangle(low = [1., 1., 1., 1.], high = [2., 2., 2., 2.])
input_vars = [:th1, :th2, :u1, :u2]
control_vars = [:T1, :T2]
dynamics = double_pend_dynamics_overt
update_rule = double_pend_update_rule
#network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/controller_simple_double_pend.nnet"
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/controller_complex_double_pend.nnet"

last_layer_activation = Id()
out_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file,
                               input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in out_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

n_sim = 1000000
out_sets_simulated = monte_carlo_simulate(double_pend_dynamics, network_file, Id(),
                                          input_set_0, n_sim, n_timesteps, dt)
fig = plot_output_sets(out_sets)
fig = plot_output_sets(out_sets_simulated, fig=fig, color=:red)

include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
include("example_dynamics.jl")
using LazySets

dt = 0.1
N_OVERT = 2
n_timesteps = 5
input_set_0 = Hyperrectangle(low = [1., 1., 1., 1.], high = [2., 2., 2., 2.])
input_vars = [:x1, :x2, :x3, :x4]
control_vars = [:c1, :c2]
dynamics, update_rule = car_dynamics_overt, car_update_rule
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_simple_car.nnet"
last_layer_activation = Id()
out_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_OVERT)

for s in out_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

n_sim = 1000000
out_sets_simulated = monte_carlo_simulate(car_dynamics, network_file, Id(),
                                          input_set_0, n_sim, n_timesteps, dt)
fig = plot_output_sets(out_sets)
fig = plot_output_sets(out_sets_simulated, fig=fig, color=:red)


# range_dict = Dict(:c1 => [-0.52, 0.83], :c2 => [-0.38, 0.96], :x1 => [-1., 1.],
#                   :x2 => [-1., 1.], :x3 => [-2., 2.], :x4 => [-2., 2.])
#
# v1 = :(0.833*tan(c2))
# v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))
#
# #v2 = :($(v1_oA.output) - $(v1_oA.output)^3/3 + $(v1_oA.output)^5/5 - $(v1_oA.output)^7/7)
# v2 = :(atan($(v1_oA.output)))
# v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))
#
# v3 = :($(v2_oA.output) + x3)
# v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))
#
# v4 = :(cos($(v3_oA.output)))
# v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))
#
# v5 = :(sin($(v3_oA.output)))
# v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))
#
# v6 = :(x4*$(v4_oA.output))
# v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))
#
# v7 = :(x4*$(v5_oA.output))
# v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))
#
# v8 = :(0.667*x4*sin($(v2_oA.output)))
# v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)
# range_dict = merge(range_dict, Dict(v8_oA.output=> v8_oA.output_range))
#
# oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA])
# mip_model = OvertMIP(oA_out)
#
# """
# # adding controller
# """
# # controller_expr1 = :(T1== th1*0.5 + u1*0.25)
# # controller_expr2 = :(T2== th2*0.5 + u2*0.25)
# # eq_2_mip(controller_expr1, mip_model)
# # eq_2_mip(controller_expr2, mip_model)
#
# input_set = Hyperrectangle(low =  [range_dict[:x1][1], range_dict[:x2][1], range_dict[:x3][1], range_dict[:x4][1]],
#                            high = [range_dict[:x1][2], range_dict[:x2][2], range_dict[:x3][2], range_dict[:x4][2]])
# control_input_vars = [get_mip_var(v, mip_model) for v in [:x1, :x2, :x3, :x4]]
# control_output_vars = [get_mip_var(v, mip_model) for v in [:c1, :c2]]
# network_file = "MIP_stuff/controller_simple_double_pend.nnet"
# controller_bound = add_controller_constraints(mip_model.model, network_file,
#                                                                 input_set,
#                                                                 control_input_vars,
#                                                                 control_output_vars,
#                                                                 #last_layer_activation= ReLU()
#                                                                 )
# @assert controller_bound.center[1] - controller_bound.radius[1] >= range_dict[:c1][1]
# @assert controller_bound.center[2] - controller_bound.radius[2] >= range_dict[:c2][1]
# @assert controller_bound.center[1] + controller_bound.radius[1] <= range_dict[:c1][2]
# @assert controller_bound.center[2] + controller_bound.radius[2] <= range_dict[:c2][2]
#
#
# # additing next_state
# dt = 0.1
# dx1 = v6_oA.output
# dx2 = v7_oA.output
# dx4 = v8_oA.output
# integration_map = Dict(:x1 => dx1, :x2 => dx2, :x3 => :c2, :x4 => dx4)
#
# using Dates
# println("begins at: ", Dates.format(now(), "HH:MM"))
# for (v, dv) in pairs(integration_map)
#     next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
#
#     @objective(mip_model.model, Min, next_v)
#     JuMP.optimize!(mip_model.model)
#     min_next_v = objective_value(mip_model.model)
#     @objective(mip_model.model, Max, next_v)
#     JuMP.optimize!(mip_model.model)
#     max_next_v = objective_value(mip_model.model)
#
#     println("next $v, min= $min_next_v, max=$max_next_v")
# end
#
# println("ends at: ", Dates.format(now(), "HH:MM"))

#
# next_u1 = mip_model.vars_dict[:u1] + dt*mip_model.vars_dict[du1]
# next_u2 = mip_model.vars_dict[:u2] + dt*mip_model.vars_dict[du2]
#
# @objective(mip_model.model, Min, next_th1)
# optimize!(mip_model.model)
# min_next_th1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_th1)
# optimize!(mip_model.model)
# max_next_th1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_th2)
# optimize!(mip_model.model)
# min_next_th2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_th2)
# optimize!(mip_model.model)
# max_next_th2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_u1)
# optimize!(mip_model.model)
# min_next_u1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_u1)
# optimize!(mip_model.model)
# max_next_u1 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Min, next_u2)
# optimize!(mip_model.model)
# min_next_u2 = objective_value(mip_model.model)
#
# @objective(mip_model.model, Max, next_u2)
# optimize!(mip_model.model)
# max_next_u2 = objective_value(mip_model.model)

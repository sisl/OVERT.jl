
OVERAPPROX_PATH = "/home/amaleki/Dropbox/stanford/Python/OverApprox"
include("$OVERAPPROX_PATH/OverApprox/src/overapprox_nd_relational.jl")
include("$OVERAPPROX_PATH/OverApprox/src/overt_parser.jl")
include("$OVERAPPROX_PATH/MIP_stuff/overt_to_mip.jl")
include("$OVERAPPROX_PATH/MIP_stuff/read_net.jl")

using LazySets

function single_pend_dynamics(range_dict, N_overt)
    v1 = :(T + sin(th) - 0.2*dth)
    v1_oA = overapprox_nd(v1, range_dict; N=N_overt)
    return v1_oA, [v1_oA.output]
end

function single_pend_update_rule(input_vars, overt_output_vars)
    ddth = overt_output_vars[1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

function find_controller_bound(network_file, input_set, last_layer_activation)
    network = read_nnet(network_file; last_layer_activation=last_layer_activation)
    bounds = get_bounds(network, input_set)
    return bounds[end]
end

function one_timestep_query(dynamics, network_file, input_set, input_vars, control_vars, last_layer_activation, dt, N_overt)
    range_dict = Dict{Symbol,Array{Float64,1}}()
    for i = 1:length(input_vars)
        range_dict[input_vars[i]] = [input_set.center[i] - input_set.radius[i],
                                     input_set.center[i] + input_set.radius[i]]
    end

    cntr_bound = find_controller_bound(network_file, input_set, last_layer_activation)
    for i = 1:length(control_vars)
        range_dict[control_vars[i]] = [cntr_bound.center[i] - cntr_bound.radius[i],
                                       cntr_bound.center[i] + cntr_bound.radius[i]]
    end

    # call overt
    oA, oA_vars = single_pend_dynamics(range_dict, N_overt)
    mip_model = OvertMIP(oA)

    # read controller
    mip_control_input_vars = [get_mip_var(v, mip_model) for v in input_vars]
    mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars]
    controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)

    # get integration map
    integration_map = single_pend_update_rule(input_vars, oA_vars)

    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)
    for v in input_vars
        dv = integration_map[v]
        next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
        @objective(mip_model.model, Min, next_v)
        JuMP.optimize!(mip_model.model)
        push!(lows, objective_value(mip_model.model))
        @objective(mip_model.model, Max, next_v)
        JuMP.optimize!(mip_model.model)
        push!(highs, objective_value(mip_model.model))
    end
    output_set = Hyperrectangle(low=lows, high=highs)
    return output_set
end

function many_timestep_query(n_timesteps, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_overt)
    input_set = input_set_0
    all_sets = [input_set]
    for i = 1:n_timesteps
        output_set = one_timestep_query(dynamics, network_file, input_set, input_vars,
                                        control_vars, last_layer_activation, dt, N_overt)
        input_set = output_set
        push!(all_sets, input_set)
    end
    return all_sets
end




input_set_0 = Hyperrectangle(low = [-1., -2.], high = [1., 2.])
input_vars = [:th, :dth]
control_vars = [:T]
dynamics = single_pend_dynamics
network_file, last_layer_activation = "OverApprox/MIP_stuff/controller_simple_single_pend.nnet", Id()
all_sets = many_timestep_query(5, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, 0.1, 5)

for s in all_sets
    for j = 1:length(s.radius)
        println(s.center[j]-s.radius[j], "  ", s.center[j]+s.radius[j])
    end
    println()
end

# control_input_vars = [get_mip_var(v, mip_model) for v in [:th, :dth]]
# control_output_vars = [get_mip_var(v, mip_model) for v in [:T]]
# network_file = "MIP_stuff/controller_simple_single_pend.nnet"
# controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, control_input_vars, control_output_vars)
# @assert controller_bound.center[1] - controller_bound.radius[1] > range_dict[:T][1]
# @assert controller_bound.center[1] + controller_bound.radius[1] < range_dict[:T][2]
#
#
#
# # additing next_state
#
#
# dt = 0.1
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

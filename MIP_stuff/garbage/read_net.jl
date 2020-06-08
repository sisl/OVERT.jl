using JuMP
using MathProgBase.SolverInterface
using GLPK
using LazySets
using Parameters
using Interpolations
using Gurobi

NEURAL_VERIFICATION_PATH = "/home/amaleki/Downloads/NeuralVerification.jl"

include("$NEURAL_VERIFICATION_PATH/src/utils/activation.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/network.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/problem.jl")
include("$NEURAL_VERIFICATION_PATH/src/utils/util.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/constraints.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/objectives.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/utils/variables.jl")
include("$NEURAL_VERIFICATION_PATH/src/optimization/mipVerify.jl")
include("$NEURAL_VERIFICATION_PATH/src/reachability/maxSens.jl")

function add_controller_constraints(model, network_nnet_address, input_set, input_vars, output_vars; last_layer_activation=Id())
    network = read_nnet(network_nnet_address, last_layer_activation=last_layer_activation)
    neurons = init_neurons(model, network)
    deltas = init_deltas(model, network)
    bounds = get_bounds(network, input_set)
    encode_network!(model, network, neurons, deltas, bounds, BoundedMixedIntegerLP())
    @constraint(model, input_vars .== neurons[1])  # set inputvars
    @constraint(model, output_vars .== neurons[end])  # set outputvars

    return bounds[end]
end

function find_controller_bound(network_file, input_set, last_layer_activation)
    network = read_nnet(network_file; last_layer_activation=last_layer_activation)
    bounds = get_bounds(network, input_set)
    return bounds[end]
end

function one_timestep_query(dynamics, update_rule, network_file, input_set, input_vars,
                             control_vars, last_layer_activation, dt, N_overt)
    range_dict = Dict{Symbol, Array{Float64,1}}()
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
    oA, oA_vars = dynamics(range_dict, N_overt)
    mip_model = OvertMIP(oA)

    # read controller
    mip_control_input_vars = [get_mip_var(v, mip_model) for v in input_vars]
    mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars]
    controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)

    mip_summary(mip_model.model)

    # get integration map
    integration_map = update_rule(input_vars, control_vars, oA_vars)

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

function many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_overt)
    input_set = input_set_0
    all_sets = [input_set]
    for i = 1:n_timesteps
        t1 = Dates.now()
        output_set = one_timestep_query(dynamics, update_rule, network_file, input_set, input_vars,
                                        control_vars, last_layer_activation, dt, N_overt)
        t2 = Dates.now()
        println("timestep $i computed in $((t2-t1).value/1000)")
        input_set = output_set
        push!(all_sets, input_set)
    end
    return all_sets
end



function mip_summary(model)

    MathOptInterface = MOI
    const_types = list_of_constraint_types(model)
    l_lin = 0
    l_bin = 0

    println("="^50)
    println("="^18 * " mip summary " * "="^19)
    println("="^50)
    for i = 1:length(const_types)
        var = const_types[i][1]
        const_type = const_types[i][2]
        l = length(all_constraints(model, var, const_type))
        #println("there are $l constraints of type $const_type with variables type $var.")
        if const_type != MathOptInterface.ZeroOne
            l_lin += l
        else
            l_bin += l
        end
    end
    #println("="^50)
    println("there are $l_lin linear constraints and $l_bin binary constraints.")
    println("="^50)
end

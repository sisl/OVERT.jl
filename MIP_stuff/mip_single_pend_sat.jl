include("../OverApprox/src/overapprox_nd_relational.jl")
include("../OverApprox/src/overt_parser.jl")
include("overt_to_mip.jl")
include("read_net.jl")
include("example_dynamics.jl")

using LazySets

dt = 0.1
N_OVERT = 10
input_set = Hyperrectangle(low = [1., 1.], high = [2., 2.])
unsafe_set = Hyperrectangle(low = [3.0, 1.8], high = [4.0, 2.5])
input_vars = [:th, :dth]
control_vars = [:T]
dynamics = single_pend_dynamics_overt
update_rule = single_pend_update_rule
network_file = "/home/amaleki/Dropbox/stanford/Python/OverApprox/MIP_stuff/nnet_files/controller_simple_single_pend.nnet"
last_layer_activation = Id()



function setup_model(dynamics, input_set, network_file, last_layer_activation, N_OVERT,
                     input_vars, control_vars)
   # building mip_model
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
   oA, oA_vars = dynamics(range_dict, N_OVERT)
   mip_model = OvertMIP(oA)

   # read controller
   mip_control_input_vars = [get_mip_var(v, mip_model) for v in input_vars]
   mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars]
   controller_bound = add_controller_constraints(mip_model.model, network_file, input_set,
                              mip_control_input_vars, mip_control_output_vars)

   return mip_model, oA_vars
end


function many_timestep_query_with_counter_example(update_rule, dynamics, network_file,
    input_set_0, unsafe_set, input_vars, control_vars, last_layer_activation, dt, N_overt)

    δ = 0.01
    all_sets = [input_set_0]
    counter_example_list = []
    counter_example = nothing

    while isnothing(counter_example)
       mip_model, oA_vars = setup_model(dynamics, all_sets[end], network_file, last_layer_activation, N_OVERT,
                                  input_vars, control_vars)
       integration_map = update_rule(input_vars, control_vars, oA_vars)
       for i = 1:length(input_vars)
          v = input_vars[i]
          dv = integration_map[v]
          next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
          v_lb = unsafe_set.center[i] - unsafe_set.radius[i]
          v_ub = unsafe_set.center[i] + unsafe_set.radius[i]
          @constraint(mip_model.model, next_v ≤ v_ub)
          @constraint(mip_model.model, next_v ≥ v_lb)
       end
       # check feasibility
       #@objective(mip_model.model, Max, mip_model.vars_dict[input_vars[2]] + dt*mip_model.vars_dict[integration_map[input_vars[2]]])
       JuMP.optimize!(mip_model.model)
       status = termination_status(mip_model.model)
       if status == MOI.OPTIMAL
          counter_example = [value(mip_model.vars_dict[v]) for v in input_vars]
          pushfirst!(counter_example_list, counter_example)
          violation = [value(mip_model.vars_dict[v]) + dt*value(mip_model.vars_dict[integration_map[v]]) for v in input_vars]
          println("time step $(length(all_sets)): counter example found $(counter_example)")
          println("violation = $(violation)")
       elseif status == MOI.INFEASIBLE
          counter_example = nothing
          output_set = one_timestep_query(dynamics, update_rule, network_file, all_sets[end], input_vars,
                                          control_vars, last_layer_activation, dt, N_overt)
          push!(all_sets, output_set)
          println(output_set)
       else
          throw("Not Optimal or infeasible. Some problem here")
       end
   end

    # while is_intersection_empty(input_set, unsafe_set)
    #     output_set = one_timestep_query(dynamics, update_rule, network_file, input_set, input_vars,
    #                                     control_vars, last_layer_activation, dt, N_overt)
    #     input_set = output_set
    #     push!(all_sets, input_set)
    #     println(input_set)
    # end


    for i = length(all_sets):-1:1
         unsafe_set = Hyperrectangle(low=counter_example .- δ, high=counter_example .+ δ)
         mip_model, oA_vars = setup_model(dynamics, all_sets[i], network_file, last_layer_activation,
                                         N_OVERT, input_vars, control_vars)
         integration_map = update_rule(input_vars, control_vars, oA_vars)
         obj = 0
         for j = 1:length(input_vars)
            v = input_vars[j]
            dv = integration_map[v]
            next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]

            v_lb = all_sets[i].center[j] - all_sets[i].radius[j]
            v_ub = all_sets[i].center[j] + all_sets[i].radius[j]
            v_next_lb = unsafe_set.center[j] - unsafe_set.radius[j]
            v_next_ub = unsafe_set.center[j] + unsafe_set.radius[j]
            @constraint(mip_model.model, mip_model.vars_dict[v] ≤ v_ub)
            @constraint(mip_model.model, mip_model.vars_dict[v] ≥ v_lb)
            @constraint(mip_model.model, next_v ≤ v_next_ub)
            @constraint(mip_model.model, next_v ≥ v_next_lb)

            ## obj = Min abs(v - (0.5*(l+u))
            ## abs(x) = relu(x) + relu(-x)
            # c, r =  all_sets[i].center[j],  all_sets[i].radius[j]
            # x_tmp = mip_model.vars_dict[v] - c
            # relu1 = get_mip_aux_var(mip_model)
            # relu2 = get_mip_aux_var(mip_model)
            # a_obj = get_mip_aux_var(mip_model, binary=true)
            #
            # # relu(x)
            # @constraint(mip_model.model, relu1 ≥ x_tmp)
            # @constraint(mip_model.model, relu1 ≥ 0)
            # @constraint(mip_model.model, relu1 ≤ a_obj * r)
            # @constraint(mip_model.model, relu1 ≤ x_tmp + (1 - a_obj) * r)
            #
            # # relu(-x)
            # @constraint(mip_model.model, relu2 ≥ -x_tmp)
            # @constraint(mip_model.model, relu2 ≥ 0)
            # @constraint(mip_model.model, relu2 ≤ r * (1 - a_obj))
            # @constraint(mip_model.model, relu2 ≤ -x_tmp + a_obj * r)
            # obj += relu1 + relu2
            # abs_var = get_mip_aux_var(mip_model)
            # @constraint(mip_model.model, abs_var ≤ (mip_model.vars_dict[input_vars[j]] - c)^2)
            # obj += abs_var
         end
         #obj = mip_model.vars_dict[input_vars[1]]
         #obj = mip_model.vars_dict[input_vars[1]] + mip_model.vars_dict[input_vars[2]]
         #obj = (mip_model.vars_dict[input_vars[1]] - all_sets[i].center[1])^2 + (mip_model.vars_dict[input_vars[2]] - all_sets[i].center[2])^2
         #@objective(mip_model.model, Min, obj)
         println(objective_function(mip_model.model))
         JuMP.optimize!(mip_model.model)
         if termination_status(mip_model.model) ≠ MOI.OPTIMAL
            throw("something went wrong. optimization status is $(termination_status(mip_model.model))")
         else
            counter_example = [value(mip_model.vars_dict[v]) for v in input_vars]
            println("time step $(i-1), counter_example= $counter_example")
            pushfirst!(counter_example_list, counter_example)
         end
      end
      return counter_example_list
end



# function rr()
#    output_sets = [input_set]
#    mip_models = []
#
#    while true
#       mip_model, oA_vars = setup_model(output_sets[end], network_file, last_layer_activation, N_OVERT,
#                            input_vars, control_vars)
#       mip_summary(mip_model.model)
#       integration_map = update_rule(input_vars, control_vars, oA_vars)
#
#       for i = 1:length(input_vars)
#          v = input_vars[i]
#          dv = integration_map[v]
#          next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
#          v_lb = unsafe_set.center[i] - unsafe_set.radius[i]
#          v_ub = unsafe_set.center[i] + unsafe_set.radius[i]
#          @constraint(mip_model.model, next_v ≤ v_ub)
#          @constraint(mip_model.model, next_v ≥ v_lb)
#       end
#
#       # check feasibility
#       JuMP.optimize!(mip_model.model)
#       status = termination_status(mip_model.model)
#       if status == MOI.OPTIMAL
#          counter_example = [value(mip_model.vars_dict[v]) for v in input_vars]
#       elseif status == MOI.INFEASIBLE
#          counter_example = nothing
#       else
#          raise("Not Optimal or infeasible. Some problem here")
#       end
#
#
#
#       if isnothing(counter_example)
#          mip_model, oA_vars = setup_model(output_sets[end], network_file, last_layer_activation, N_OVERT,
#                               input_vars, control_vars)
#          integration_map = update_rule(input_vars, control_vars, oA_vars)
#          push!(mip_models, mip_model)
#          lows = Array{Float64}(undef, 0)
#          highs = Array{Float64}(undef, 0)
#          for v in input_vars
#               dv = integration_map[v]
#               next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
#               @objective(mip_model.model, Min, next_v)
#               JuMP.optimize!(mip_model.model)
#               push!(lows, objective_value(mip_model.model))
#               @objective(mip_model.model, Max, next_v)
#               JuMP.optimize!(mip_model.model)
#               push!(highs, objective_value(mip_model.model))
#          end
#          output_set = Hyperrectangle(low=lows, high=highs)
#          push!(output_sets, output_set)
#       else
#          println("counter example found")
#          for i = length(output_sets):-1:1
#             mip_model = setup_model()
#             for (v, val) in zip(input_vars, counter_example)
#                @constraint(mip_model.model, mip_model.vars_dict[v] == val)
#             end
#             JuMP.optimize!(mip_model.model)
#             if termination_status(mip_model.model) ≠ MOI.OPTIMAL
#                raise("backtrack failed")
#             else
#                counter_example = [value(mip_model.vars_dict[v]) for v in input_vars]
#             end
#          end
#       end
#    end
# end

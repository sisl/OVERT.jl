function setup_overt_and_controller(dynamics, network_file, input_set, input_vars,
                          control_vars, last_layer_activation, N_overt)

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
   return mip_model, oA_vars
end


function find_next_state_bounds(dynamics, update_rule, network_file, input_set, input_vars,
                             control_vars, last_layer_activation, dt, N_overt)
    mip_model, oA_vars = setup_overt_and_controller(dynamics, network_file, input_set, input_vars,
                     control_vars, last_layer_activation, N_overt)

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


function find_counter_example(dynamics, update_rule, network_file, input_set, output_set,
                           input_vars, control_vars, last_layer_activation, dt, N_overt; δ₀=0)

    mip_model, oA_vars = setup_overt_and_controller(dynamics, network_file, input_set, input_vars,
                   control_vars, last_layer_activation, N_overt)

    integration_map = update_rule(input_vars, control_vars, oA_vars)
    for i = 1:length(input_vars)
       v = input_vars[i]
       dv = integration_map[v]
       next_v =  mip_model.vars_dict[v] + dt*mip_model.vars_dict[dv]
       v_lb = input_set.center[i] - input_set.radius[i]
       v_ub = input_set.center[i] + input_set.radius[i]
       next_v_lb = output_set.center[i] - output_set.radius[i] * (1 - δ₀)
       next_v_ub = output_set.center[i] + output_set.radius[i] * (1 - δ₀)
       @constraint(mip_model.model, mip_model.vars_dict[v] ≤ v_ub)
       @constraint(mip_model.model, mip_model.vars_dict[v] ≥ v_lb)
       @constraint(mip_model.model, next_v ≤ next_v_ub)
       @constraint(mip_model.model, next_v ≥ next_v_lb)
    end
    # check feasibility

    #@objective(mip_model.model, Max, mip_model.vars_dict[input_vars[2]] + dt*mip_model.vars_dict[integration_map[input_vars[2]]])
    JuMP.optimize!(mip_model.model)
    status = termination_status(mip_model.model)
    counter_example = nothing
    target = nothing
    if status == MOI.OPTIMAL
       counter_example = [value(mip_model.vars_dict[v]) for v in input_vars]
       target = [value(mip_model.vars_dict[v] + dt*mip_model.vars_dict[integration_map[v]]) for v in input_vars]
       println("counter example found: $(counter_example)")
    elseif status == MOI.INFEASIBLE
       println("counter example was NOT found.")
    else
       throw("Not Optimal or infeasible. Some problem here")
    end
    return counter_example, target
end

function satisfiability_search(dynamics, update_rule, network_file, input_set_0, unsafe_set,
                           input_vars, control_vars, last_layer_activation, dt, N_overt;
                           max_timesteps=10, δ=0.001)
   all_ce = []
   all_sets = [input_set_0]
   input_set = input_set_0

   ce = nothing
   violation = nothing
   for itr = 1:max_timesteps
      ce, violation = find_counter_example(dynamics, update_rule, network_file, input_set, unsafe_set,
                                 input_vars, control_vars, last_layer_activation, dt, N_overt)
      output_set = find_next_state_bounds(dynamics, update_rule, network_file, input_set, input_vars,
                                   control_vars, last_layer_activation, dt, N_overt)
      println(output_set)
      push!(all_sets, output_set)
      if isnothing(ce)
         input_set = output_set
      else
         break
      end
   end


   if isnothing(ce)
      println("you're safe in the next $max_timesteps timesteps.")
      return all_sets, all_ce, violation
   else
      pushfirst!(all_ce, ce)
   end

   for i = 1:length(input_vars)
      @assert ce[i] ≥ all_sets[end-1].center[i] - all_sets[end-1].radius[i]
      @assert ce[i] ≤ all_sets[end-1].center[i] + all_sets[end-1].radius[i]
      @assert violation[i] ≥ all_sets[end].center[i] - all_sets[end].radius[i]
      @assert violation[i] ≤ all_sets[end].center[i] + all_sets[end].radius[i]
   end


   for i = length(all_sets)-2:-1:1
      goal_set = Hyperrectangle(low = ce .- δ, high = ce .+ δ)
      input_set = all_sets[i]
      ce = find_counter_example(dynamics, update_rule, network_file, input_set, goal_set,
                                 input_vars, control_vars, last_layer_activation, dt, N_overt)[1]
      @assert !isnothing(ce)
      for j = 1:length(input_vars)
         @assert ce[j] ≥ all_sets[i].center[j] - all_sets[i].radius[j]
         @assert ce[j] ≤ all_sets[i].center[j] + all_sets[i].radius[j]
      end
      pushfirst!(all_ce, ce)
   end

   return all_sets, all_ce, violation
end



function plot_satisfiability(unsafe_set, all_sets, all_counter_examples, network_file,
   exact_dynamics, violation; idx=[1,2])
    fig = plot_output_sets(all_sets; idx=idx)
    fig = plot_output_sets([unsafe_set], fig=fig, fillalpha=0.7, fill=:red, linewidth=3)

    if all_counter_examples == []
      return fig
    end

    xc = [c[idx[1]] for c in all_counter_examples]
    yc = [c[idx[2]] for c in all_counter_examples]
    push!(xc, violation[idx[1]])
    push!(yc, violation[idx[2]])

    fig = plot!(xc, yc, marker=:o, markersize=5, linewidth=2, linecolor=:yellow)

    xd = [all_counter_examples[1][idx[1]]]
    yd = [all_counter_examples[1][idx[2]]]

    s = all_counter_examples[1]
    controller = read_nnet(network_file)
    for i = 1:length(xc)-1
        u = compute_output(controller, s)
        s = exact_dynamics(s, u)
        push!(xd, s[idx[1]])
        push!(yd, s[idx[2]])
    end

    fig = plot!(xc, yc, marker=:s, markersize=5, linewidth=2, linecolor=:blue)
    return fig
end



function get_range_dict(input_set, input_vars, control_vars, network_file, last_layer_activation)
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
   return range_dict
end

function symbolic_bound(n_timesteps, update_rule, dynamics, dynamics_timed, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_overt)
   all_sets = many_timestep_query(n_timesteps, update_rule, dynamics, network_file, input_set_0, input_vars, control_vars, last_layer_activation, dt, N_overt)

   oA_list = Array{OverApproximation}(undef, 0)
   oA_vars_list = []
   for i = 1:n_timesteps+1
      input_vars_timed = [Meta.parse("$(v)_$i") for v in input_vars]
      control_vars_timed = [Meta.parse("$(v)_$i") for v in control_vars]
      input_set = all_sets[i]
      range_dict = get_range_dict(input_set, input_vars_timed, control_vars_timed, network_file, last_layer_activation)
      oA, oA_vars = dynamics_timed(range_dict, N_overt, i)
      push!(oA_list, oA)
      push!(oA_vars_list, oA_vars)
   end

   oA_tot = add_overapproximate(oA_list)
   mip_model = OvertMIP(oA_tot)

   last_time_step_vars = []
   for i = 1:n_timesteps
      oA_vars_now = oA_vars_list[i]
      input_vars_now = [Meta.parse("$(v)_$i") for v in input_vars]
      input_vars_next= [Meta.parse("$(v)_$(i+1)") for v in input_vars]
      control_vars_now = [Meta.parse("$(v)_$i") for v in control_vars]

      integration_map = update_rule(input_vars_now, control_vars_now, oA_vars_now)
      for j = 1:length(input_vars)
         v = input_vars_now[j]
         dv = integration_map[v]
         next_v = input_vars_next[j]

         v_mip = mip_model.vars_dict[v]
         dv_mip = mip_model.vars_dict[dv]
         next_v_mip = mip_model.vars_dict[next_v]
         if i == n_timesteps
            push!(last_time_step_vars, next_v_mip)
         end
         @constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip)
     end
   end


   input_vars_last = [Meta.parse("$(v)_$n_timesteps") for v in input_vars]
   lows = Array{Float64}(undef, 0)
   highs = Array{Float64}(undef, 0)
   for v in last_time_step_vars
      @objective(mip_model.model, Min, v)
      JuMP.optimize!(mip_model.model)
      push!(lows, objective_value(mip_model.model))
      @objective(mip_model.model, Max, v)
      JuMP.optimize!(mip_model.model)
      push!(highs, objective_value(mip_model.model))
   end
   all_sets_symbolic = Hyperrectangle(low=lows, high=highs)
   return all_sets, all_sets_symbolic
end

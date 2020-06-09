using JuMP
using MathProgBase.SolverInterface
using GLPK
using LazySets
using Parameters
using Interpolations
using Gurobi
using Dates

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

"""
----------------------------------------------
read controller
----------------------------------------------
"""

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


"""
----------------------------------------------
reasoning and solving queries
----------------------------------------------
"""

function setup_overt_and_controller_constraints(query::OvertQuery, input_set::Hyperrectangle; t_idx::Union{Int64, Nothing}=nothing)
   """
	This function reads overt query, computes the dictionary range, runs overt,
      reads the controller nnet file,  and finally setup the overt
      and controller constraints in form of a JuMP MIP model.
   inputs:
   - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
   - t_idx: this is for timed dynamics, when the symbolic version is called.
            if t_idx is an integer, a superscript will be added to all state and
            control variable, indicating the timestep is a symbolic represenation.
            default is nothing, which does not do the timed dynamics.
   outputs:
   - mip_model: OvertMIP file that contain the MIP formulation,
   - oA: all OvertApproximation objects used in the dynamics,
   - oA_vars: all Overt output variables.
	"""

   # read attributes
   dynamics = query.problem.overt_dynamics
   input_vars = query.problem.input_vars
   control_vars = query.problem.control_vars

   network_file = query.network_file
   last_layer_activation = query.last_layer_activation
   N_overt = query.N_overt

   # adding subscripts if variables are timed
   if !isnothing(t_idx)
      input_vars = [Meta.parse("$(v)_$t_idx") for v in input_vars]
      control_vars = [Meta.parse("$(v)_$t_idx") for v in control_vars]
   end

   # setup overt range dictionary
   range_dict = Dict{Symbol, Array{Float64,1}}()
   for i = 1:length(input_vars)
     range_dict[input_vars[i]] = [input_set.center[i] - input_set.radius[i],
                                   input_set.center[i] + input_set.radius[i]]
   end

   # read controler and add bounds to range dictionary
   cntr_bound = find_controller_bound(network_file, input_set, last_layer_activation)
   for i = 1:length(control_vars)
     range_dict[control_vars[i]] = [cntr_bound.center[i] - cntr_bound.radius[i],
                                     cntr_bound.center[i] + cntr_bound.radius[i]]
   end

   # call overt and setup overtMIP
   oA, oA_vars = dynamics(range_dict, N_overt, t_idx)
   mip_model = OvertMIP(oA)

   # add controller to mip
   mip_control_input_vars = [get_mip_var(v, mip_model) for v in input_vars]
   mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars]
   controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)

   mip_summary(mip_model.model)
   return mip_model, oA, oA_vars
end

function one_timestep_concretization(query::OvertQuery, input_set::Hyperrectangle; t_idx::Union{Int64, Nothing}=nothing)
   """
	This function computes the next reachable set.
   inputs:
   - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
   - t_idx: this is for timed dynamics, when the symbolic version is called.
            if t_idx is an integer, a superscript will be added to all state and
            control variable, indicating the timestep is a symbolic represenation.
            default is nothing, which does not do the timed dynamics.
   outputs:
   - output_set: a hyperrectangle for the output reachable set
   - oA: all OvertApproximation objects used in the dynamics,
   - oA_vars: all Overt output variables.
	"""
	# reading some query attributes
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	dt = query.dt

	# setup input and control variables if timed
	if !isnothing(t_idx)
      input_vars = [Meta.parse("$(v)_$t_idx") for v in input_vars]
      control_vars = [Meta.parse("$(v)_$t_idx") for v in control_vars]
   end

   # setup overt and controller constraints
   mip_model, oA, oA_vars = setup_overt_and_controller_constraints(query, input_set; t_idx=t_idx)

   # get integration map
   integration_map = query.problem.update_rule(input_vars, control_vars, oA_vars)

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
   return output_set, oA, oA_vars
end

function many_timestep_concretization(query::OvertQuery, input_set_0::Hyperrectangle; timed::Bool=false)
   """
	This function computes the next n reachable sets using concretization.
   inputs:
   - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
   - t_idx: this is for timed dynamics, when the symbolic version is called.
            if t_idx is an integer, a superscript will be added to all state and
            control variable, indicating the timestep is a symbolic represenation.
            default is nothing, which does not do the timed dynamics.
   outputs:
   - all_sets: an array of hyperrectangle of all reachable sets, starting from the init set
   - oA: an array of OvertApproximation objects used in each timestep of concretization,
   - oA_vars: an array of Overt output variables used in each timestep of concretization,
	"""
    input_set = copy(input_set_0)
    all_sets = [input_set_0]
    all_oA = Array{OverApproximation}(undef, 0)
    all_oA_vars = []
    for i = 1:query.ntime + 1
        t1 = Dates.now()
        if timed
           output_set, oA, oA_vars = one_timestep_concretization(query, input_set; t_idx=i)
        else
           output_set, oA, oA_vars = one_timestep_concretization(query, input_set)
        end
        t2 = Dates.now()
        println("timestep $i computed in $((t2-t1).value/1000)")
        input_set = output_set
        push!(all_sets, output_set)
        push!(all_oA, oA)
        push!(all_oA_vars, oA_vars)
    end

    return all_sets,  all_oA, all_oA_vars
end


function symbolic_bound(query, input_set)
   """
	This function computes the reachable set after n timestep symbolically.
   inputs:
   - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
   outputs:
   - all_sets: an array of hyperrectangle of all reachable sets, starting from the init set
               computed with concretization
   - all_sets_symbolic: a hyperrectangle for the reachable set at t=n, computed symbolically.
	"""
	# read some attributes
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	network_file = query.network_file
	ntime = query.ntime
	update_rule = query.problem.update_rule
	dt = query.dt

	# setup all overt constraints via bounds found by conceretization
	all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
	oA_tot = add_overapproximate(all_oA)
	mip_model = OvertMIP(oA_tot)

	# add controller to mip
	for i = 1:ntime+1
		input_set        = all_sets[i]
		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
	end

   mip_summary(mip_model.model)

   last_time_step_vars = []
   for i = 1:ntime
      oA_vars_now = all_oA_vars[i]
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
         if i == ntime
            push!(last_time_step_vars, next_v_mip)
         end
         @constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip)
     end
   end


   input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
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
   return all_sets[1:end-1], all_sets_symbolic
end

"""
----------------------------------------------
monte carlo simulation
----------------------------------------------
"""

function monte_carlo_simulate(query::OvertQuery, input_set::Hyperrectangle; n_sim::Int64=1000000)
	"""
	Running monte carlo simulations for comparison.
   inputs:
   - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
	- n_sim: number of mc simulations
   outputs:
   - output_sets: output sets at all times
	- xvec: simulation results.
	"""

	# unpacking query attributes
	dynamics_func = query.problem.true_dynamics
	controller_nnet_address  = query.network_file
	last_layer_activation = query.last_layer_activation
	ntime = query.ntime
	dt = query.dt
	n_states = length(input_set.center)
	min_x = [[Inf64  for n = 1:n_states] for m = 1:ntime]
	max_x = [[-Inf64 for n = 1:n_states] for m = 1:ntime]
	controller = read_nnet(controller_nnet_address, last_layer_activation=last_layer_activation)
	xvec = zeros(n_sim, ntime, n_states)
	for i = 1:n_sim
	  x  = rand(n_states)
	  x .*= input_set.radius * 2
	  x .+= input_set.center - input_set.radius
	  for j = 1:ntime
	      u = compute_output(controller, x)
	      dx = dynamics_func(x, u)
	      x = x + dx*dt
	      min_x[j] = min.(x, min_x[j])
	      max_x[j] = max.(x, max_x[j])
	      xvec[i, j, :] = x
	  end
	end

	output_sets = [input_set]
	for (m1, m2) in zip(min_x, max_x)
	  println(m1, m2)
	  push!(output_sets, Hyperrectangle(low=m1, high=m2))
	end
return output_sets, xvec
end

"""
----------------------------------------------
plotting
----------------------------------------------
"""

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function plot_output_sets(output_sets; idx=[1,2], fig=nothing, linewidth=3,
    linecolor=:black, linestyle=:solid, fillalpha=0, fill=:red)

    p = isnothing(fig) ? plot() : fig
    for s in output_sets
        w = s.radius[idx[1]] * 2
        h = s.radius[idx[2]] * 2
        x = s.center[idx[1]] - s.radius[idx[1]]
        y = s.center[idx[2]] - s.radius[idx[2]]
        #plot!(rectangle(w,h,x,y), fillalpha=0.0, kwargs)
        plot!(rectangle(w,h,x,y), fillalpha=fillalpha, fill=fill, legend=nothing,
                   linewidth=linewidth, linecolor=linecolor, linestyle=linestyle)
        xlabel!("x_$(idx[1])")
        ylabel!("x_$(idx[2])")
    end
    return p
end

function plot_output_hist(data, ntime; fig=nothing, idx=[1,2], nbins=:notspecified)
    p = isnothing(fig) ? plot() : fig
    x = data[:, ntime, idx[1]]
    y = data[:, ntime, idx[2]]
    if nbins == :notspecified
        fig = histogram2d!(x, y, legend=false)
    else
        fig = histogram2d!(x, y, nbins=nbins, legend=false)
    end
    return fig
end

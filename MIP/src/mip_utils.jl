using JuMP
using MathProgBase.SolverInterface
using GLPK
using LazySets
using Parameters
using Interpolations
using Gurobi
using Dates
using MathOptInterface

include("../nv/utils/activation.jl")
include("../nv/utils/network.jl")
include("../nv/utils/problem.jl")
include("../nv/utils/util.jl")
include("../nv/optimization/utils/constraints.jl")
include("../nv/optimization/utils/objectives.jl")
include("../nv/optimization/utils/variables.jl")
#include("nv/optimization/mipVerify.jl")
include("../nv/reachability/maxSens.jl")

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

   # read controller and add bounds to range dictionary
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
    for i = 1:query.ntime
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

function symbolic_satisfiability(query, input_set, target_set; unsat_problem=false)
	"""
	Checks whether a property P is satisfied at timesteps 1 to n symbolically.
	inputs:
	- query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
	- target_set: Hyperrectangle for the targe set of variables.
	- unsat_problem: if true, it solves an unsatisfiability problem. i.e. the first
	                timestep at which state does not include in target set.
	outputs:
	- status (sat, unsat, error), values if sat, statistics
	"""
	n = query.ntime
	SATus, vals, stats = "", Dict(), Dict() # "init" values...
	problem_type = unsat_problem ? "unsat" : "sat"
	for i = 1:n
		println("checking timestep ", i)
		query.ntime = i
		SATus, vals, stats = symbolic_satisfiability_nth(query, input_set, target_set)
		if SATus == problem_type
			println("Property violated at timestep $i")
			return SATus, vals, stats
	 	elseif SATus == "error"
		 	throw("some error occured at timestep $i")
		end
   end
   println("Property holds for $n timesteps.")
   return SATus, vals, stats
end

function symbolic_satisfiability_nth(query, input_set, target_set)
   """
	This function computes whether property P is satisfiable at timestep n symbolically.
    inputs:
    - query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
	- target_set: Hyperrectangle for the targe set of variables.
    outputs:
    - status (sat, unsat, error), values if sat, statistics
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
    for i = 1:ntime
		input_set        = all_sets[i]
		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
    end

    mip_summary(mip_model.model)

    for i = 1:ntime - 1
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

			@constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip) # euler integration. we should probably make room for more flexible integration schemes.
		end
    end

	# no objective makes it is a feasibility problem
	timestep_nplus1_vars = []
	input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
	control_vars_last = [Meta.parse("$(v)_$ntime") for v in control_vars]
	oA_vars_last = all_oA_vars[ntime]
	integration_map = update_rule(input_vars_last, control_vars_last, oA_vars_last)
	for (i, v) in enumerate(input_vars_last)
		v_mip = mip_model.vars_dict[v]
		dv = integration_map[v]
		dv_mip = mip_model.vars_dict[dv]
		next_v_mip = v_mip + dt * dv_mip
		push!(timestep_nplus1_vars, next_v_mip)
		v_min = target_set.center[i] - target_set.radius[i]
		v_max = target_set.center[i] + target_set.radius[i]
		@constraint(mip_model.model, next_v_mip >= v_min)
		@constraint(mip_model.model, next_v_mip <= v_max)
	end

	JuMP.optimize!(mip_model.model)




	if termination_status(mip_model.model) == MathOptInterface.OPTIMAL
		# optimal
		vals = value.(timestep_nplus1_vars)
		stats = Dict()
		for i = 1:ntime
			input_vars_now = [Meta.parse("$(v)_$i") for v in input_vars]
			tmp_dict = Dict((v, value(mip_model.vars_dict[v])) for v in input_vars_now)
			stats = merge(stats, tmp_dict)
		end

		return "sat", vals, stats
	elseif termination_status(mip_model.model) == MathOptInterface.INFEASIBLE
		# infeasible
		return "unsat", Dict(), Dict()
	else
  		return "error", Dict(), Dict()
	end
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
	for i = 1:ntime
		input_set        = all_sets[i]
		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
	end

    mip_summary(mip_model.model)

	# connect outputs of timestep i to inputs of timestep i-1
	for i = 1:ntime - 1
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

			@constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip)
     	end
   	end

	# optimize for the output of timestep ntime.
	lows = Array{Float64}(undef, 0)
	highs = Array{Float64}(undef, 0)
	input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
	control_vars_last = [Meta.parse("$(v)_$ntime") for v in control_vars]
	oA_vars_last = all_oA_vars[ntime]
	integration_map = update_rule(input_vars_last, control_vars_last, oA_vars_last)
	for v in input_vars_last
	   	v_mip = mip_model.vars_dict[v]
		dv = integration_map[v]
		dv_mip = mip_model.vars_dict[dv]
		next_v_mip = v_mip + dt * dv_mip
		@objective(mip_model.model, Min, next_v_mip)
		JuMP.optimize!(mip_model.model)
		push!(lows, objective_value(mip_model.model))
		@objective(mip_model.model, Max, next_v_mip)
		JuMP.optimize!(mip_model.model)
		push!(highs, objective_value(mip_model.model))
   	end
	all_sets_symbolic = Hyperrectangle(low=lows, high=highs)
	return all_sets, all_sets_symbolic
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


function plot_output_sets_pgfplot(output_sets; idx=[1,2], fig=nothing, linewidth=3,
    linecolor=:black, linestyle=:solid, fillalpha=0, fill=:red, labels=nothing)

    if isnothing(fig)
		fig = PGFPlots.Axis(style="width=10cm, height=10cm")
	end

	if !isnothing(labels)
		fig.xlabel = labels[1]
		fig.ylabel = labels[2]
	end


	line_style = "$linestyle, $linecolor, very thick, mark=none"
    for s in output_sets
		s1, s2 = s.center[idx[1]], s.center[idx[2]]
		r1, r2 = s.radius[idx[1]], s.radius[idx[2]]
		push!(fig, PGFPlots.Plots.Linear([s1-r1, s1+r1], [s2+r2, s2+r2], style=line_style))
		push!(fig, PGFPlots.Plots.Linear([s1-r1, s1+r1], [s2-r2, s2-r2], style=line_style))
		push!(fig, PGFPlots.Plots.Linear([s1-r1, s1-r1], [s2-r2, s2+r2], style=line_style))
		push!(fig, PGFPlots.Plots.Linear([s1+r1, s1+r1], [s2-r2, s2+r2], style=line_style))
    end

    return fig
end

function plot_output_hist_pgfplot(data, ntime; fig=nothing, idx=[1,2],
	     inner_points=false, labels=nothing)
    if isnothing(fig)
		fig = PGFPlots.Axis(style="width=10cm, height=10cm")
	end

    # x = data[:, ntime, idx[1]]
    # y = data[:, ntime, idx[2]]
	# push!(fig, PGFPlots.Plots.Histogram2(x, y, density=true,
	#                                    colormap=PGFPlots.ColorMaps.Named("Jet")))

	points = data[:, ntime, idx]
	if inner_points
		dx = floor(size(points)[1] / 10000)
		dx = max(1, Int(dx))
		pp = PGFPlots.Plots.Scatter(points[1:dx:end, 1], points[1:dx:end, 2], style="mark=o, orange")
		push!(fig, pp)
	end

	border_idx = chull(points).vertices
	p = PGFPlots.Plots.Linear(points[border_idx, 1], points[border_idx, 2],
	                         style="solid, orange, line width=3pt, mark=none")
	push!(fig, p)

	if !isnothing(labels)
		fig.xlabel = labels[1]
		fig.ylabel = labels[2]
	end

    return fig
end

using JuMP
using MathProgBase.SolverInterface
using GLPK
using LazySets
using Parameters
using Interpolations
using Gurobi
using Dates
using MathOptInterface
#using PGFPlots

include("../nv/utils/activation.jl")
include("../nv/utils/network.jl")
include("../nv/utils/problem.jl")
include("../nv/utils/util.jl")
include("../nv/optimization/utils/constraints.jl")
include("../nv/optimization/utils/objectives.jl")
include("../nv/optimization/utils/variables.jl")
include("../nv/reachability/maxSens.jl")
include("logic.jl")

"""
Type Definitions.
"""
mutable struct MyHyperrect
	low::Array
	high::Array
end
MyHyperrect(; low=[], high=[]) = MyHyperrect(low,high)
low(h::MyHyperrect) = h.low
high(h::MyHyperrect) = h.high


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
reasoning and solving concrete queries
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
	- mip_model: OvertMIP file that contain the MIP formulation of all overt and
	             controller constraints.
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

function solve_for_reachability(mip_model::OvertMIP, query::OvertQuery,
	oA_vars::Array{Symbol, 1}, t_idx::Union{Int64, Nothing}; get_meas=false)
	"""
	this function sets up states in the future timestep and optimizes that.
	this will give a minimum and maximum on each of the future timesteps.
	the final answer is a hyperrectangle.
	inputs:
	- mip_model: OvertMIP object that contains all overt and controller constraints.
	- query: OvertQuery
	- oA_vars: output variables of overt.
	- t_idx: this is for timed dynamics, when the symbolic version is called.
	        if t_idx is an integer, a superscript will be added to all state and
	        control variable, indicating the timestep is a symbolic represenation.
	        default is nothing, which does not do the timed dynamics.
	outputs:
	- reacheable_set: smallest hyperrectangle that contains the reachable set.
	"""

	# inputs of reachable set hyperrectangle are initialized
	lows = Array{Float64}(undef, 0)
	highs = Array{Float64}(undef, 0)

	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	update_rule = query.problem.update_rule
	dt = query.dt

	if !isnothing(t_idx)
		input_vars_last = [Meta.parse("$(v)_$t_idx") for v in input_vars]
		control_vars_last = [Meta.parse("$(v)_$t_idx") for v in control_vars]
		integration_map = update_rule(input_vars_last, control_vars_last, oA_vars)
	else
		input_vars_last = input_vars
		integration_map = query.problem.update_rule(input_vars, control_vars, oA_vars)
	end

	# setup the future state and optimize.
	timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
	for v in input_vars_last
	   	v_mip = mip_model.vars_dict[v]
		dv = integration_map[v]
		dv_mip = mip_model.vars_dict[dv]
		next_v_mip = v_mip + dt * dv_mip
		push!(timestep_nplus1_vars, next_v_mip)
		@objective(mip_model.model, Min, next_v_mip)
		JuMP.optimize!(mip_model.model)
		@assert termination_status(mip_model.model) == MathOptInterface.OPTIMAL
		push!(lows, objective_bound(mip_model.model))
		@debug "Objective value for MIN is: $(objective_value(mip_model.model)) and objective bound is $(objective_bound(mip_model.model))"
		@objective(mip_model.model, Max, next_v_mip)
		JuMP.optimize!(mip_model.model)
		@assert termination_status(mip_model.model) == MathOptInterface.OPTIMAL
		push!(highs, objective_bound(mip_model.model))
		@debug "Objective value for MAX is: $(objective_value(mip_model.model)) and objective bound is $(objective_bound(mip_model.model))"
	end
	# get the hyperrectangle.
	reacheable_set = Hyperrectangle(low=lows, high=highs)

	# measurement model code,
	# compute reachable set for measurements
	if query.problem.measurement_model != [] && get_meas
		@debug "Computing measurement model reachable sets."
		meas_lows = Array{Float64}(undef, 0)
		meas_highs = Array{Float64}(undef, 0)
		# deal with computing reachable set for measurements
		for measurement_matrix_row in query.problem.measurement_model 
			@objective(mip_model.model, Min, sum(measurement_matrix_row.*timestep_nplus1_vars))
			JuMP.optimize!(mip_model.model)
			@assert termination_status(mip_model.model) == MathOptInterface.OPTIMAL
			push!(meas_lows, objective_bound(mip_model.model))
			@objective(mip_model.model, Max, sum(measurement_matrix_row.*timestep_nplus1_vars))
			JuMP.optimize!(mip_model.model)
			@assert termination_status(mip_model.model) == MathOptInterface.OPTIMAL
			push!(meas_highs, objective_bound(mip_model.model))
		end
		meas_reachable_set = Hyperrectangle(low=meas_lows, high=meas_highs)
	else
		meas_reachable_set = nothing
	end
	
	return reacheable_set, meas_reachable_set
end

function one_timestep_concretization(query::OvertQuery, input_set::Hyperrectangle; t_idx::Union{Int64, Nothing}=nothing, get_meas=false)
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

	# setup overt and controller constraints
	mip_model, oA, oA_vars = setup_overt_and_controller_constraints(query, input_set; t_idx=t_idx)

	# solve for reacheability
	output_set, measurement_output_set = solve_for_reachability(mip_model, query, oA_vars, t_idx; get_meas=get_meas) 
	return output_set, measurement_output_set, oA, oA_vars 
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
	all_meas_sets = []
    all_oA = Array{OverApproximation}(undef, 0)
    all_oA_vars = []
    for i = 1:query.ntime
        t1 = Dates.now()
        if timed
		   output_set, meas_output_set, oA, oA_vars = one_timestep_concretization(query, input_set; t_idx=i, get_meas=true)
		   #output_set, oA, oA_vars = one_timestep_concretization(query, input_set; t_idx=i)
        else
		   output_set, meas_output_set, oA, oA_vars = one_timestep_concretization(query, input_set; get_meas=true)
		   #output_set, oA, oA_vars = one_timestep_concretization(query, input_set)
        end
        t2 = Dates.now()
        println("timestep $i computed in $((t2-t1).value/1000) seconds")
        input_set = output_set
		push!(all_sets, output_set)
		push!(all_meas_sets, meas_output_set)
        push!(all_oA, oA)
        push!(all_oA_vars, oA_vars)
    end

	return all_sets, all_meas_sets, all_oA, all_oA_vars
	#return all_sets, all_oA, all_oA_vars
end

"""
----------------------------------------------
reasoning and solving symbolic queries, basic functions
----------------------------------------------
"""

function setup_mip_with_overt_constraints(query::OvertQuery, input_set::Hyperrectangle)
	"""
	this function computes the query using concrete method for the purpose
		of setting up all overt constraints.
		Ideally, all overt and controller constrains at all timesteps should be computed
		and a mip_model object is constructed for each time step. However, JuMP.model
		does not support merging different models.
		The work around this is to compute all overt constraints alltogether and
		then combine them with controller constraints.
	inputs:
    - query: OvertQuery
 	- input_set: Hyperrectangle for the initial set of variables.
    outputs:
    - mip_model: OvertMIP file that contain the MIP formulation, and overt constraints.
	             Notice that controller constraints are NOT yet added.
	- oA: all OvertApproximation objects used in the dynamics,
	- oA_vars: all Overt output variables.
	"""
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	network_file = query.network_file
	ntime = query.ntime
	update_rule = query.problem.update_rule
	dt = query.dt

	# setup all overt constraints via bounds found by conc retization
	all_sets, all_meas_sets, all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
	#all_sets, all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
	oA_tot = add_overapproximate(all_oA)
	mip_model = OvertMIP(oA_tot)
	return mip_model, all_sets, all_meas_sets, all_oA_vars
	#return mip_model, all_sets, all_oA_vars		
end

function add_controllers_constraints!(mip_model, query::OvertQuery, all_sets)
	"""
	This function add controller constraints to mip_model.
	mip_model is changed in place.
	inputs:
	-mip_model: OvertMIP file that contain the MIP formulation, and overt constraints.
	             but does not yet contain controller constraints.
	- query: OvertQuery
	- all_sets: a list of all reachable sets at all timesteps computed using the concrete approach.
	"""
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	network_file = query.network_file
	ntime = query.ntime

	# add controller to mip
	for i = 1:ntime
		input_set        = all_sets[i]
		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
	end
end

function match_io!(mip_model, query, all_oA_vars)
	"""
	This function connects output of current timestep to the intputs of future
	timestep, via introducing new constraints in the mip formulation.
	mip_model is changed in place.
	inputs:
	-mip_model: OvertMIP file that contain the MIP formulation, and overt constraints.
	             but does not yet contain controller constraints.
	- query: OvertQuery
	- all_oA_vars:an array of Overt output variables used in each timestep
	"""
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	ntime = query.ntime
	update_rule = query.problem.update_rule
	dt = query.dt

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
end

"""
----------------------------------------------
symbolic queries, reachability.
----------------------------------------------
"""

function symbolic_reachability(query::OvertQuery, input_set::Hyperrectangle)
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
	# setup all overt cosntraints
	mip_model, all_sets, all_meas_sets, all_oA_vars = setup_mip_with_overt_constraints(query, input_set)
	#mip_model, all_sets, all_oA_vars = setup_mip_with_overt_constraints(query, input_set)

	# read neural network and add controller constraints
	add_controllers_constraints!(mip_model, query, all_sets)

	# mip summary of constaints
    mip_summary(mip_model.model)

	# connect outputs of timestep i to inputs of timestep i-1
	match_io!(mip_model, query, all_oA_vars)

	# optimize for the output of timestep ntime.
	set_symbolic, measurement_set_symbolic = solve_for_reachability(mip_model, query, all_oA_vars[end], query.ntime) 
	return all_sets, set_symbolic, all_meas_sets, measurement_set_symbolic
end

"""
----------------------------------------------
symbolic queries, reachability with splitting.
----------------------------------------------
"""

function split_hyperrectangle(input_set::Hyperrectangle, splits_idx)
	"""
 	This function splits the input_set into halves based on indices given in splits_idx
 	- input_set: Hyperrectangle for the initial set of variables.
	- splits_idx: indeces for splitting.
    outputs:
    - input_sets_splitted: an array of hyperrectangles which are splits of
	the original input_set.
 	"""
	if length(splits_idx) == 0
		return input_set
	else
		input_sets_splitted = []
		idx = splits_idx[1]
		center = input_set.center
		radius = input_set.radius

		left_center = copy(center)
		left_center[idx] -= radius[idx] / 2
		left_radius = copy(radius)
		left_radius[idx] /= 2
		left = Hyperrectangle(left_center, left_radius)

		right_center = copy(center)
		right_center[idx] += radius[idx] / 2
		right_radius = copy(radius)
		right_radius[idx] /= 2
		right = Hyperrectangle(right_center, right_radius)

		push!(input_sets_splitted, split_hyperrectangle(left, splits_idx[2:end]))

		push!(input_sets_splitted, split_hyperrectangle(right, splits_idx[2:end]))
	end
	input_sets_splitted = vcat(input_sets_splitted...)
	return input_sets_splitted
end

function symbolic_reachability_with_splitting(query::OvertQuery, input_sets::Array{Any, 1}, splits_idx::Array{Int, 1})
	all_concrete_sets = []
	all_symbolic_sets = []
	all_concrete_meas_sets = []
	all_symbolic_meas_sets = []
	for input_set in input_sets
		concrete_sets, symbolic_set, concrete_meas_sets, symbolic_meas_set = symbolic_reachability_with_splitting(query, input_set, splits_idx)
		# concrete_sets,  symbolic_set = symbolic_reachability_with_splitting(query, input_set, splits_idx)
		push!(all_concrete_sets, concrete_sets)
		push!(all_symbolic_sets, symbolic_set)
		push!(all_concrete_meas_sets, concrete_meas_sets)
		push!(all_symbolic_meas_sets, symbolic_meas_set)
	end
	return all_concrete_sets, all_symbolic_sets, all_concrete_meas_sets, all_symbolic_meas_sets
	#return all_concrete_sets, all_symbolic_sets
end

function symbolic_reachability_with_splitting(query::OvertQuery, input_set::Hyperrectangle, splits_idx::Array{Int, 1})
	"""
 	This function splits the input_set into halves based on indices given in splits_idx
		and then computes the reachable set after n timestep symbolically
    inputs:
    - query: OvertQuery
 	- input_set: Hyperrectangle for the initial set of variables.
	- splits_idx: indeces for splitting.
    outputs:
    - all_sets: an array of hyperrectangle of all reachable sets, starting from the init set
                computed with concretization
    - all_sets_symbolic: a hyperrectangle for the reachable set at t=n, computed symbolically.
 	"""
	input_sets_splitted = split_hyperrectangle(input_set, splits_idx)
	all_concrete_sets = []
	all_symbolic_sets = []
	all_concrete_meas_sets = []
	all_symbolic_meas_sets = []
	for this_set in input_sets_splitted
		concrete_sets, symbolic_set, concrete_meas_sets, symbolic_meas_set = symbolic_reachability(query, this_set)
		#concrete_sets, symbolic_set = symbolic_reachability(query, this_set)
		push!(all_concrete_sets, concrete_sets)
		push!(all_symbolic_sets, symbolic_set)
		push!(all_concrete_meas_sets, concrete_meas_sets)
		push!(all_symbolic_meas_sets, symbolic_meas_set)
	end
	return all_concrete_sets, all_symbolic_sets, all_concrete_meas_sets, all_symbolic_meas_sets
	#return all_concrete_sets, all_symbolic_sets
end

function symbolic_reachability_with_concretization(query::OvertQuery,
	input_set::Hyperrectangle, concretize_every::Union{Int, Array{Int, 1}})
   """
	This function computes the reachable set after n timestep symbolically by
		concretizing after every concretize_every timesteps.

   inputs:
   - query: OvertQuery
   - input_set: Hyperrectangle for the initial set of variables.
   - concretize_every: concretization period.
   outputs:
   - all_sets: an array of hyperrectangle of all reachable sets, starting from the init set
               computed with concretization
   - all_sets_symbolic: a hyperrectangle for the reachable set at t=n, computed symbolically.
	"""
	ntime = query.ntime # number of timesteps
	if isa(concretize_every, Int)
		@assert ntime % concretize_every == 0
		n_loops = Int(query.ntime / concretize_every)
		concretize_every = [concretize_every for i in 1:n_loops]
	end


	all_concrete_sets = []
	all_symbolic_sets = []
	all_concrete_meas_sets = []
	all_symbolic_meas_sets = []
	this_set = copy(input_set)
	for n in concretize_every
		query.ntime = n
		concrete_sets, symbolic_set, concrete_meas_sets, symbolic_set_meas = symbolic_reachability(query, this_set) # pass query and input set
		# concrete_sets, symbolic_set = symbolic_reachability(query, this_set) # pass query and input set
		push!(all_concrete_sets, concrete_sets)
		push!(all_symbolic_sets, symbolic_set)
		push!(all_concrete_meas_sets, concrete_meas_sets)
		push!(all_symbolic_meas_sets, symbolic_meas_set)
		this_set = copy(symbolic_set)
	end

	query.ntime = ntime
	return all_concrete_sets, all_symbolic_sets, all_concrete_meas_sets, all_symbolic_meas_sets
	# return all_concrete_sets, all_symbolic_sets
end

function symbolic_reachability_with_concretization_with_splitting(query::OvertQuery,
	input_sets::Array{Any, 1},
	concretize_every::Union{Int, Array{Int, 1}},
	split_idx::Array{Int, 1})

	all_concrete_sets = []
	all_symbolic_sets = []
	all_concrete_meas_sets = []
	all_symbolic_meas_sets = []
	for input_set in input_sets
		concrete_sets, concrete_meas_sets, symbolic_set, symbolic_set_meas = symbolic_reachability_with_concretization_with_splitting(query, input_set, concretize_every_split_idx)
		push!(all_concrete_sets, concrete_sets)
		push!(all_symbolic_sets, symbolic_set)
		push!(all_concrete_meas_sets, concrete_meas_sets)
		push!(all_symbolic_meas_sets, symbolic_meas_set)
	end
	return all_concrete_sets, all_symbolic_sets, all_concrete_meas_sets, all_symbolic_meas_sets
end

function symbolic_reachability_with_concretization_with_splitting(query::OvertQuery,
	input_set::Hyperrectangle,
	concretize_every::Union{Int, Array{Int, 1}},
	split_idx::Array{Int, 1})

	ntime = query.ntime
	if isa(concretize_every, Int)
		@assert ntime % concretize_every == 0
		n_loops = Int(query.ntime / concretize_every)
		concretize_every = [concretize_every for i in 1:n_loops]
	end

	all_concrete_sets = []
	all_symbolic_sets = []
	all_concrete_meas_sets = []
	all_symbolic_meas_sets = []
	this_set = copy(input_set)
	idx = 0
	for n in concretize_every
		idx += 1
		println("concretize step: $idx")
		query.ntime = n
		concrete_sets, concrete_meas_sets, symbolic_set, symbolic_set_meas = symbolic_reachability_with_splitting(query, this_set, split_idx)
		push!(all_concrete_sets, concrete_sets)
		push!(all_symbolic_sets, symbolic_set)
		push!(all_concrete_meas_sets, concrete_meas_sets)
		push!(all_symbolic_meas_sets, symbolic_meas_set)
		this_set = copy(symbolic_set)
	end

	query.ntime = ntime
	return all_concrete_sets, all_symbolic_sets, all_concrete_meas_sets, all_symbolic_meas_sets
end

"""
----------------------------------------------
symbolic queries, satisfiability (feasibility).
----------------------------------------------
"""

function add_feasibility_constraints!(mip_model, query, oA_vars, target_constraints)
	"""
	this function adds the constraints associated with checking whether future states
	intersect with the target set or not.
	the mip_model is changed in place.
	inputs:
	- mip_model: OvertMIP object that contains all overt and controller constraints.
	- query: OvertQuery
	- oA_vars: output variables of overt.
	- target_constraints: EITHER a hyperrectangle for the target set OR a list of Constraint objects
	outputs:
	- timestep_nplus1_vars: a list of mip_variables assocaited with the states of
							next timestep this will be used for computing the
							point of feasibility (counter example)

	"""
	input_vars = query.problem.input_vars
	control_vars = query.problem.control_vars
	ntime = query.ntime
	update_rule = query.problem.update_rule
	dt = query.dt

	# setup variables current timestep
	timestep_nplus1_vars = []
	input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
	control_vars_last = [Meta.parse("$(v)_$ntime") for v in control_vars]
	integration_map = update_rule(input_vars_last, control_vars_last, oA_vars)

	# setup variables at next timestep 
	timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
	for (i, v) in enumerate(input_vars_last)
		v_mip = mip_model.vars_dict[v]
		dv = integration_map[v]
		dv_mip = mip_model.vars_dict[dv]
		next_v_mip = v_mip + dt * dv_mip
		push!(timestep_nplus1_vars, next_v_mip)
	end

	# if a measurement model is present, check variables on measurement model
	if query.problem.measurement_model != []
		@debug "Checking feasibility of property on measurement variables, not state variables."
		output_vars = GenericAffExpr{Float64,VariableRef}[]
		for measurement_matrix_row in query.problem.measurement_model 
			push!(output_vars, sum(measurement_matrix_row.*timestep_nplus1_vars))
		end
	else
		@debug "Checking feasibility of property state variables."
		output_vars = timestep_nplus1_vars
	end

	# Add constraints for feasibility.
	add_output_constraints!(target_constraints, mip_model.model, output_vars)
		
	return timestep_nplus1_vars
end

function add_output_constraints!(target_set, model::JuMP.Model, vars::Array)
	# Version for hyperrectangle outputs ::Union{Hyperrectangle,MyHyperrect,InfiniteHyperrectangle}
	for (i, v) in enumerate(vars)
		v_min = low(target_set)[i] 
		v_max = high(target_set)[i]
		if isfinite(v_min)
			@constraint(model, v >= v_min)
		end
		if isfinite(v_max)
			@constraint(model, v <= v_max)
		end
	end
end

function add_output_constraints!(constraint::Constraint, model::JuMP.Model, vars::Array)
	# Constraint contains the coeffs, relation, and scalar, assuming the variables are in 
	# state order
	# construct a constraint: a^T x R s
	# where a is the coefficient vector, x is the variable vector, R is the relation and s is the scalar

	# reshape
	x = reshape(vars, (length(vars), 1))
	c = reshape(constraint.coeffs, (1, length(constraint.coeffs)))
	if constraint.relation == :(<=)
		@constraint(model, (c*x)[1] <= constraint.scalar)
	elseif constraint.relation == :(>=)
		@constraint(model, (c*x)[1] >= constraint.scalar)
	elseif constraint.relation == :(==)
		@constraint(model, (c*x)[1] == constraint.scalar)
	end

end

# function symbolic_satisfiability_nth(query::OvertQuery, input_set::Hyperrectangle,
# 	target_set::Hyperrectangle)
# 	"""
# 	This function computes the reachable set after n timestep symbolically.
# 	inputs:
# 	- query: OvertQuery
# 	- input_set: Hyperrectangle for the initial set of states.
# 	- target_set: Hyperrectangle for the target set of states.
# 	outputs:
# 	- status: status of query which can be sat, unsat or error,
# 	- vals: if sat, returns the counter example at timestep n+1. else returns empty dictionary.
# 	- stats: if sat, returns the counter example at timestep 1 to n. else returns empty dictionary.
# 	"""
# 	vals, stats = Dict(), Dict()
#
# 	# setup all overt cosntraints
#
# 	mip_model, all_sets, all_oA_vars = setup_mip_with_overt_constraints(query, input_set)
#
# 	# read neural network and add controller constraints
# 	add_controllers_constraints!(mip_model, query, all_sets)
#
# 	# mip summary of constaints
#     mip_summary(mip_model.model)
#
# 	# connect outputs of timestep i to inputs of timestep i-1
# 	match_io!(mip_model, query, all_oA_vars)
#
# 	# add feasibility constraints that n+1 timestep intersects target set.
# 	timestep_nplus1_vars = add_feasibility_constraints!(mip_model, query, all_oA_vars[end], target_set)
#
# 	JuMP.optimize!(mip_model.model)
# 	if termination_status(mip_model.model) == MathOptInterface.OPTIMAL
# 		# optimal
# 		status = "sat"
# 		vals = value.(timestep_nplus1_vars)
# 		stats = Dict()
# 		for i = 1:query.ntime
# 			input_vars_now = [Meta.parse("$(v)_$i") for v in query.problem.input_vars]
# 			tmp_dict = Dict((v, value(mip_model.vars_dict[v])) for v in input_vars_now)
# 			stats = merge(stats, tmp_dict)
# 		end
# 	elseif termination_status(mip_model.model) == MathOptInterface.INFEASIBLE
# 		# infeasible
# 		status = "unsat"
# 	else
#   		status = "error"
# 	end
# 	return status, vals, stats
# end
#
#
# function symbolic_satisfiability(query::OvertQuery, input_set::Hyperrectangle, target_set::Hyperrectangle; unsat_problem::Bool=false)
# 	"""
# 	Checks whether a property P is satisfied at timesteps 1 to n symbolically.
# 	inputs:
# 	- query: OvertQuery
# 	- input_set: Hyperrectangle for the initial set of variables.
# 	- target_set: Hyperrectangle for the targe set of variables.
# 	- unsat_problem: if true, it solves an unsatisfiability problem. i.e. the first
# 	                timestep at which state does not include in target set.
# 	outputs:
# 	- status: status of query which can be sat, unsat or error,
# 	- vals: if sat, returns the counter example at timestep n+1. else returns empty dictionary.
# 	- stats: if sat, returns the counter example at timestep 1 to n. else returns empty dictionary.
# 	"""
# 	n = query.ntime
# 	SATus, vals, stats = "", Dict(), Dict() # "init" values...
# 	problem_type = unsat_problem ? "unsat" : "sat"
# 	for i = 1:n
# 		println("checking timestep ", i)
# 		query.ntime = i
# 		SATus, vals, stats = symbolic_satisfiability_nth(query, input_set, target_set)
# 		if SATus == problem_type
# 			println("Property violated at timestep $i")
# 			return SATus, vals, stats
# 	 	elseif SATus == "error"
# 		 	throw("some error occured at timestep $i")
# 		end
#    end
#    println("Property holds for $n timesteps.")
#    return SATus, vals, stats
# end


function symbolic_satisfiability_nth(query::OvertQuery, input_set::Hyperrectangle,
	target_set, all_sets, all_oA, all_oA_vars; threads=0)
	"""
	This function computes the reachable set after n timestep symbolically.
	inputs:
	- query: OvertQuery
	- input_set: Hyperrectangle for the initial set of states.
	- target_set: Hyperrectangle for the target set of states.
	outputs:
	- status: status of query which can be sat, unsat or error,
	- vals: if sat, returns the counter example at timestep n+1. else returns empty dictionary.
	- stats: if sat, returns the counter example at timestep 1 to n. else returns empty dictionary.
	"""
	vals, stats = Dict(), Dict()
	ntime = query.ntime
	@assert ntime == length(all_oA)

	# combine overt cosntraints
	oA_tot = add_overapproximate(all_oA)
	mip_model = OvertMIP(oA_tot, threads=threads)

	# read neural network and add controller constraints
	add_controllers_constraints!(mip_model, query, all_sets)

	# mip summary of constaints
    mip_summary(mip_model.model)

	# connect outputs of timestep i to inputs of timestep i-1
	match_io!(mip_model, query, all_oA_vars)

	# add feasibility constraints on n+1 timestep OR to measurement that is function of n+1 time states
	timestep_nplus1_vars = add_feasibility_constraints!(mip_model, query, all_oA_vars[end], target_set)

	JuMP.optimize!(mip_model.model)
	if termination_status(mip_model.model) == MathOptInterface.OPTIMAL
		# optimal
		status = "sat"
		vals = value.(timestep_nplus1_vars)
		stats = Dict()
		for i = 1:query.ntime
			input_vars_now = [Meta.parse("$(v)_$i") for v in query.problem.input_vars]
			tmp_dict = Dict((v, value(mip_model.vars_dict[v])) for v in input_vars_now)
			stats = merge(stats, tmp_dict)
		end
	elseif termination_status(mip_model.model) == MathOptInterface.INFEASIBLE
		# infeasible
		status = "unsat"
	else
  		status = "error"
	end
	return status, vals, stats
end


# function symbolic_satisfiability(query::OvertQuery, input_set::Hyperrectangle, target_set::Hyperrectangle; unsat_problem::Bool=false)
# 	"""
# 	Checks whether a property P is satisfied at timesteps 1 to n symbolically.
# 	inputs:
# 	- query: OvertQuery
# 	- input_set: Hyperrectangle for the initial set of variables.
# 	- target_set: Hyperrectangle for the targe set of variables.
# 	- unsat_problem: if true, it solves an unsatisfiability problem. i.e. the first
# 	                timestep at which state does not include in target set.
# 	outputs:
# 	- status: status of query which can be sat, unsat or error,
# 	- vals: if sat, returns the counter example at timestep n+1. else returns empty dictionary.
# 	- stats: if sat, returns the counter example at timestep 1 to n. else returns empty dictionary.
# 	"""
# 	n = query.ntime
# 	SATus, vals, stats = "", Dict(), Dict() # "init" values...
# 	problem_type = unsat_problem ? "unsat" : "sat"
#
# 	all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
#
# 	for i = 1:n
# 		println("checking timestep ", i)
# 		query.ntime = i
# 		SATus, vals, stats = symbolic_satisfiability_nth(query, input_set, target_set, all_sets[1:i], all_oA[1:i], all_oA_vars[1:i])
# 		if SATus == problem_type
# 			println("Property violated at timestep $i")
# 			return SATus, vals, stats
# 	 	elseif SATus == "error"
# 		 	throw("some error occured at timestep $i")
# 		end
#    end
#    println("Property holds for $n timesteps.")
#    return SATus, vals, stats
# end


function symbolic_satisfiability(query::OvertQuery, input_set::Hyperrectangle, target_set; unsat_problem::Bool=false, after_n::Int=0, return_all=false, threads=0)
	"""
	Checks whether a property P is satisfied at timesteps 1 to n symbolically.
	inputs:
	- query: OvertQuery
	- input_set: Hyperrectangle for the initial set of variables.
	- target_set: Hyperrectangle for the targe set of variables.
	- unsat_problem: if true, it solves an unsatisfiability problem. i.e. the first
	                timestep at which state does not include in target set.
    - after_n: satisfiability is checked for n > after_n. n here is the time step number.
	           default value = 0. meaning that we start from begining.
	outputs:
	- status: status of query which can be sat, unsat or error,
	- vals: if sat, returns the counter example at timestep n+1. else returns empty dictionary.
	- stats: if sat, returns the counter example at timestep 1 to n. else returns empty dictionary.

	"""
	n = query.ntime
	if return_all
		SATii, valii, statii = [], [], []
	else
		SATus, vals, stats = "", Dict(), Dict() # "init" values...
	end
	problem_type = unsat_problem ? "unsat" : "sat"
	println("problem type: ", problem_type)

	input_set_tmp = copy(input_set)
	all_sets = [input_set]
	# all meas sets = [] # don't need to save measurement sets  for a satisfiability problem
	all_oA = Array{OverApproximation}(undef, 0)
	all_oA_vars = []

	for i = 1:n
		println("checking timestep ", i)
		output_set, meas_set, oA, oA_vars = one_timestep_concretization(query, input_set_tmp; t_idx=i, get_meas=false)
		#output_set, oA, oA_vars = one_timestep_concretization(query, input_set_tmp; t_idx=i)
		input_set_tmp = output_set
		push!(all_sets, output_set)
		# push!(all_meas_sets, meas_set) # don't need to save measurement sets  for a satisfiability problem
		push!(all_oA, oA)
		push!(all_oA_vars, oA_vars)

		if i <= after_n
			println("not yet in the timestep of interest")
			continue
		end

		query.ntime = i
		SATus, vals, stats = symbolic_satisfiability_nth(query, input_set, target_set, all_sets, all_oA, all_oA_vars, threads=threads)
		println("SATus of step ", i, " is ", SATus)
		if return_all
			push!(SATii, SATus)
			push!(valii, vals)
			push!(statii, stats)
		end
		if SATus == problem_type # todo: clean up this problem type stuff
			println("Property violated at timestep $i")
			if ~return_all
				return SATus, vals, stats
			end
	 	elseif SATus == "error"
		 	throw("some error occured at timestep $i")
		end
   	end
	if return_all
		return SATii, valii, statii
	else
		println("Property holds for $n timesteps.")
		return SATus, vals, stats
	end
end





# function symbolic_satisfiability_nth(query, input_set, target_set)
#    """
# 	This function computes whether property P is satisfiable at timestep n symbolically.
#     inputs:
#     - query: OvertQuery
# 	- input_set: Hyperrectangle for the initial set of variables.
# 	- target_set: Hyperrectangle for the targe set of variables.
#     outputs:
#     - status (sat, unsat, error), values if sat, statistics
# 	"""
# 	# read some attributes
# 	input_vars = query.problem.input_vars
# 	control_vars = query.problem.control_vars
# 	network_file = query.network_file
# 	ntime = query.ntime
# 	update_rule = query.problem.update_rule
# 	dt = query.dt
#
# 	# setup all overt constraints via bounds found by conceretization
# 	all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
# 	oA_tot = add_overapproximate(all_oA)
# 	mip_model = OvertMIP(oA_tot)
#
#     # add controller to mip
#     for i = 1:ntime
# 		input_set        = all_sets[i]
# 		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
# 		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
# 		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
# 		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
# 		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
#     end
#
#     mip_summary(mip_model.model)
#
#     for i = 1:ntime - 1
# 		oA_vars_now = all_oA_vars[i]
# 		input_vars_now = [Meta.parse("$(v)_$i") for v in input_vars]
# 		input_vars_next= [Meta.parse("$(v)_$(i+1)") for v in input_vars]
# 		control_vars_now = [Meta.parse("$(v)_$i") for v in control_vars]
#
# 		integration_map = update_rule(input_vars_now, control_vars_now, oA_vars_now)
# 		for j = 1:length(input_vars)
# 			v = input_vars_now[j]
# 			dv = integration_map[v]
# 			next_v = input_vars_next[j]
#
# 			v_mip = mip_model.vars_dict[v]
# 			dv_mip = mip_model.vars_dict[dv]
# 			next_v_mip = mip_model.vars_dict[next_v]
#
# 			@constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip) # euler integration. we should probably make room for more flexible integration schemes.
# 		end
#     end
#
# 	# no objective makes it is a feasibility problem
# 	timestep_nplus1_vars = []
# 	input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
# 	control_vars_last = [Meta.parse("$(v)_$ntime") for v in control_vars]
# 	oA_vars_last = all_oA_vars[ntime]
# 	integration_map = update_rule(input_vars_last, control_vars_last, oA_vars_last)
# 	for (i, v) in enumerate(input_vars_last)
# 		v_mip = mip_model.vars_dict[v]
# 		dv = integration_map[v]
# 		dv_mip = mip_model.vars_dict[dv]
# 		next_v_mip = v_mip + dt * dv_mip
# 		push!(timestep_nplus1_vars, next_v_mip)
# 		v_min = target_set.center[i] - target_set.radius[i]
# 		v_max = target_set.center[i] + target_set.radius[i]
# 		@constraint(mip_model.model, next_v_mip >= v_min)
# 		@constraint(mip_model.model, next_v_mip <= v_max)
# 	end
#
# 	JuMP.optimize!(mip_model.model)
#
#
#
#
# 	if termination_status(mip_model.model) == MathOptInterface.OPTIMAL
# 		# optimal
# 		vals = value.(timestep_nplus1_vars)
# 		stats = Dict()
# 		for i = 1:ntime
# 			input_vars_now = [Meta.parse("$(v)_$i") for v in input_vars]
# 			tmp_dict = Dict((v, value(mip_model.vars_dict[v])) for v in input_vars_now)
# 			stats = merge(stats, tmp_dict)
# 		end
#
# 		return "sat", vals, stats
# 	elseif termination_status(mip_model.model) == MathOptInterface.INFEASIBLE
# 		# infeasible
# 		return "unsat", Dict(), Dict()
# 	else
#   		return "error", Dict(), Dict()
# 	end
# end


# function symbolic_bound(query, input_set)
#    """
# 	This function computes the reachable set after n timestep symbolically.
#    inputs:
#    - query: OvertQuery
# 	- input_set: Hyperrectangle for the initial set of variables.
#    outputs:
#    - all_sets: an array of hyperrectangle of all reachable sets, starting from the init set
#                computed with concretization
#    - all_sets_symbolic: a hyperrectangle for the reachable set at t=n, computed symbolically.
# 	"""
# 	# read some attributes
# 	input_vars = query.problem.input_vars
# 	control_vars = query.problem.control_vars
# 	network_file = query.network_file
# 	ntime = query.ntime
# 	update_rule = query.problem.update_rule
# 	dt = query.dt
#
# 	# setup all overt constraints via bounds found by conceretization
# 	all_sets,  all_oA, all_oA_vars = many_timestep_concretization(query, input_set; timed=true)
# 	oA_tot = add_overapproximate(all_oA)
# 	mip_model = OvertMIP(oA_tot)
#
# 	# add controller to mip
# 	for i = 1:ntime
# 		input_set        = all_sets[i]
# 		input_vars_tmp   = [Meta.parse("$(v)_$i") for v in input_vars]
# 		control_vars_tmp = [Meta.parse("$(v)_$i") for v in control_vars]
# 		mip_control_input_vars  = [get_mip_var(v, mip_model) for v in input_vars_tmp]
# 		mip_control_output_vars = [get_mip_var(v, mip_model) for v in control_vars_tmp]
# 		controller_bound = add_controller_constraints(mip_model.model, network_file, input_set, mip_control_input_vars, mip_control_output_vars)
# 	end
#
#     mip_summary(mip_model.model)
#
# 	# connect outputs of timestep i to inputs of timestep i-1
# 	for i = 1:ntime - 1
# 		oA_vars_now = all_oA_vars[i]
# 		input_vars_now = [Meta.parse("$(v)_$i") for v in input_vars]
# 		input_vars_next= [Meta.parse("$(v)_$(i+1)") for v in input_vars]
# 		control_vars_now = [Meta.parse("$(v)_$i") for v in control_vars]
#
# 		integration_map = update_rule(input_vars_now, control_vars_now, oA_vars_now)
# 		for j = 1:length(input_vars)
# 			v = input_vars_now[j]
# 			dv = integration_map[v]
# 			next_v = input_vars_next[j]
#
# 			v_mip = mip_model.vars_dict[v]
# 			dv_mip = mip_model.vars_dict[dv]
# 			next_v_mip = mip_model.vars_dict[next_v]
#
# 			@constraint(mip_model.model, next_v_mip == v_mip + dt * dv_mip)
#      	end
#    	end
#
# 	# optimize for the output of timestep ntime.
# 	lows = Array{Float64}(undef, 0)
# 	highs = Array{Float64}(undef, 0)
# 	input_vars_last = [Meta.parse("$(v)_$ntime") for v in input_vars]
# 	control_vars_last = [Meta.parse("$(v)_$ntime") for v in control_vars]
# 	oA_vars_last = all_oA_vars[ntime]
# 	integration_map = update_rule(input_vars_last, control_vars_last, oA_vars_last)
# 	for v in input_vars_last
# 	   	v_mip = mip_model.vars_dict[v]
# 		dv = integration_map[v]
# 		dv_mip = mip_model.vars_dict[dv]
# 		next_v_mip = v_mip + dt * dv_mip
# 		@objective(mip_model.model, Min, next_v_mip)
# 		JuMP.optimize!(mip_model.model)
# 		push!(lows, objective_value(mip_model.model))
# 		@objective(mip_model.model, Max, next_v_mip)
# 		JuMP.optimize!(mip_model.model)
# 		push!(highs, objective_value(mip_model.model))
#    	end
# 	all_sets_symbolic = Hyperrectangle(low=lows, high=highs)
# 	return all_sets, all_sets_symbolic
# end
"""
----------------------------------------------
Checking Reachability Queries
----------------------------------------------
"""
function clean_up_sets(all_sets, symbolic_sets, conc_ints; dims=[4,5])
	# concretization intervals helps you parse which of all_sets are symbolic and which are one-step concrete. if the concretization intervals are [10,2] all_sets will contain:
	# init_set, c_t1, c_t2, ..., c_t10, s_t10, c_t11, c_t12
	# where c_t1 = concrete set at time t=1
	# s_t1 = symbolic set at time t=1
	# In constract, symbolic_sets contains:
	# s_t10, s_t12

	# in case all_sets is a set of sets:
	all_sets = vcat(all_sets...)
	sym_idx = cumsum(conc_ints)
	init_set = all_sets[1]
	dup_idx = findall(all_sets .∈ Ref(symbolic_sets)) # find indices of symbolic sets
	deleteat!(all_sets, dup_idx)
	deleteat!(all_sets, 1) # delete init set too. there should now be symidx[end] sets in all_sets, all concrete
	reachable_sets = deepcopy(all_sets)
	for (sym_interval, i) in enumerate(sym_idx)
		reachable_sets[i] = symbolic_sets[sym_interval]
	end
	# debug
	function debug_setcleanup()
		concrete_2d = get_subsets(all_sets, dims)
		reachable_2d = get_subsets(reachable_sets, dims)
		plot(concrete_2d, color="grey", title="debugging set cleanup")
		plot!(reachable_2d, color="red")
	end
	@debug debug_setcleanup()
	return init_set, reachable_sets
end

function check_avoid_set_intersection(reachable_sets, init_set, avoid_sets)
	# if there is more than one avoid set, they are intersected separately
	# check init set, but should satisfy 
	safe = true
	violations = []
	for (as_idx, as) in enumerate(avoid_sets)
		empty_intersects = [isempty(rs ∩ as) for rs in reachable_sets]
		if all(empty_intersects)
			println("Safe for avoid set $as_idx")
		else
			println("Unsafe for avoid set $as_idx. See: $empty_intersects")
			# record index of avoid set and reachable set(s) that intersected
			push!(violations, (as_idx, findall(.!empty_intersects)))
		end
		safe &= all(empty_intersects)
	end
	return safe, violations
end

function check_goal_set_achieved(reachable_sets, init_set, goal_set)
	# check if and when the goal set is check_goal_set_achieved
	# do this by checking if the reachable set is a subset of the goal set at any point in time
	achieved = [rs ⊆ goal_set for rs in reachable_sets]
	idx = findall(achieved .== true)
	if length(idx) == 0
		return false, nothing
	else
		return true, idx[1] # return first index where property is achieved!
	end
end

"""
----------------------------------------------
monte carlo simulation
----------------------------------------------
"""
function monte_carlo_one_simulate(query, x0)
	dynamics_func = query.problem.true_dynamics
	controller_nnet_address  = query.network_file
	last_layer_activation = query.last_layer_activation
	ntime = query.ntime
	dt = query.dt
	controller = read_nnet(controller_nnet_address, last_layer_activation=last_layer_activation)
	xvec = zeros(ntime+1, length(x0))
	xvec[1, :] = x0
	x = x0
	for j = 1:ntime
		u = compute_output(controller, x)
		dx = dynamics_func(x, u)
		x = x + dx*dt
		xvec[j+1, :] = x
	end
	return xvec
end


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
	x0 = zeros(n_sim, n_states)
	for i = 1:n_sim
	  x  = rand(n_states)
	  x .*= input_set.radius * 2
	  x .+= input_set.center - input_set.radius
	  x0[i, :] = x
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
return output_sets, xvec, x0
end

"""
----------------------------------------------
plotting
----------------------------------------------
"""

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function plot_output_sets(output_sets; idx=[1,2], fig=nothing, linewidth=3,
    linecolor=:black, linestyle=:solid, fillalpha=0, fill=:red)

    fig = isnothing(fig) ? Plots.plot() : fig
    for s in output_sets
        w = s.radius[idx[1]] * 2
        h = s.radius[idx[2]] * 2
        x = s.center[idx[1]] - s.radius[idx[1]]
        y = s.center[idx[2]] - s.radius[idx[2]]
        #plot!(rectangle(w,h,x,y), fillalpha=0.0, kwargs)
        Plots.plot!(rectangle(w,h,x,y), fillalpha=fillalpha, fill=fill, legend=nothing,
                   linewidth=linewidth, linecolor=linecolor, linestyle=linestyle)
        Plots.xlabel!("x_$(idx[1])")
        Plots.ylabel!("x_$(idx[2])")
    end
    return fig
end

function plot_output_hist(data, ntime; fig=nothing, idx=[1,2], nbins=:notspecified)
    fig = isnothing(fig) ? Plots.plot() : fig
    x = data[:, ntime, idx[1]]
    y = data[:, ntime, idx[2]]
    if nbins == :notspecified
        fig = Plots.histogram2d!(x, y, legend=false)
    else
        fig = Plots.histogram2d!(x, y, nbins=nbins, legend=false)
    end
    return fig
end

function plot_mc_trajectories(data0, data; fig=nothing, idx=[1,2], n_traj = 100,
	linecolor=:red, makercolor=:orange, markersize=3)
	fig = isnothing(fig) ? Plots.plot() : fig
	x0 = data0[1:n_traj, idx[1]]
	y0 = data0[1:n_traj, idx[2]]
    x = data[1:n_traj, :, idx[1]]
    y = data[1:n_traj, :, idx[2]]
	x = hcat(x0, x)
	y = hcat(y0, y)
	for i = 1:n_traj
		plot!(x[i, :], y[i, :], marker=:o, linecolor=linecolor,
		markercolor=makercolor, markersize=markersize)
	end
	return fig
end


function plot_output_sets_pgfplot(output_sets; idx=[1,2], fig=nothing, linewidth=:thick,
    linecolor=:black, linestyle=:solid, fillalpha=0, fill=:red, labels=nothing)

	if isnothing(labels)
		labels = ["\$x_$(idx[1])\$", "\$x_$(idx[2])\$"]
	end

	if isnothing(fig)
		fig = PGFPlots.Axis(style="width=10cm, height=10cm", xlabel=labels[1], ylabel=labels[2])
	end

	line_style = "$linestyle, $linecolor, $linewidth, mark=none"
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
    if isnothing(labels)
		labels = ["\$x_$(idx[1])\$", "\$x_$(idx[2])\$"]
	end

	if isnothing(fig)
		fig = PGFPlots.Axis(style="width=10cm, height=10cm", xlabel=labels[1], ylabel=labels[2])
	end

    # x = data[:, ntime, idx[1]]
    # y = data[:, ntime, idx[2]]
	# push!(fig, PGFPlots.Plots.Histogram2(x, y, density=true,
	#                                    colormap=PGFPlots.ColorMaps.Named("Jet"))

	points = data[:, ntime, idx]
	if inner_points
		dx = floor(size(points)[1] / 10000)
		dx = max(1, Int(dx))
		pp = PGFPlots.Plots.Scatter(points[1:dx:end, 1], points[1:dx:end, 2], style="mark=o, orange")
		push!(fig, pp)
	end

	border_idx = chull(points).vertices
	p = PGFPlots.Plots.Linear(points[border_idx, 1], points[border_idx, 2], style="solid, orange, line width=3pt, mark=none")
	push!(fig, p)



    return fig
end

function plot_satisfiability_pgfplot(stats, vals, query; fig=nothing, idx=[1,2], labels=nothing)
	if isnothing(labels)
		labels = ["\$x_$(idx[1])\$", "\$x_$(idx[2])\$"]
	end
	if isnothing(fig)
		fig = PGFPlots.Axis(style="width=10cm, height=10cm", xlabel=labels[1], ylabel=labels[2])
	end

	data_sat_x = Array{Float64, 1}(undef, 0)
	data_sat_y = Array{Float64, 1}(undef, 0)
    i = 1
	while true
		vx = "x$(idx[1])_$i"
		vx = Meta.parse(vx)
		if vx ∉ keys(stats)
			break
		end
		push!(data_sat_x, stats[vx])

		vy = "x$(idx[2])_$i"
		vy = Meta.parse(vy)
		push!(data_sat_y, stats[vy])
		i += 1
	end
	push!(data_sat_x, vals[idx[1]])
	push!(data_sat_y, vals[idx[2]])

	n_dim = length(query.problem.input_vars)
	x0 = [stats[Meta.parse("x$(idx[i])_1")] for i = 1:n_dim]
	data_mc = monte_carlo_one_simulate(query, x0)

	pp = PGFPlots.Plots.Linear(data_sat_x, data_sat_y, style="mark=none, orange, dashed, very thick")
	push!(fig, pp)
	pp = PGFPlots.Plots.Linear(data_mc[:, idx[1]], data_mc[:, idx[2]], style="blue")
	push!(fig, pp)
	return fig
end

# make_animation with @animate macro

function get_subsets(sets, dims)
	return [get_subset(s, dims) for s in sets]
end

# utility to extract subsets of certain dims 
# for an array of hyperrectangles (input and output type)
get_subset(h::Hyperrectangle{Float64,Array{Float64,1},Array{Float64,1}}, dims) = Hyperrectangle(low=low(h)[dims], high=high(h)[dims])

# for an array of HalfSpaces
# assumes that dimensions that are not selected are set to zero
function get_subset(h::HalfSpace{Float64,Array{Float64,1}}, dims)
	# if the indices NOT selected do NOT have coefficients == 0 in h.a, show @info? or @warn
	other_dims = setdiff([1:length(h.a)...], dims)
	other_dim_coeffs = h.a[other_dims]
	if !all(isapprox.(other_dim_coeffs, Ref(0.), atol=eps()))
		@warn "Halfspace representation is inaccurate. A new method must be implemented to adjust the b value of the halfspace to plot accurate projections."
	end
	return HalfSpace(h.a[dims], h.b)
end

function get_lims(sets, dims)
	lows = [low(h)[dims] for h in sets]
	highs = [high(h)[dims] for h in sets]
	lowest = minimum(vcat(lows'...), dims=1)
	highest = maximum(vcat(highs'...), dims=1)
	limits = collect(zip(lowest, highest))
	return limits[1], limits[2]
end

function expand_lims(lim_pair)
	low, high = lim_pair
	return (low - 0.1*abs(low), high + 0.1*abs(high))
end

function plot_reachable_sets(reachable_sets, target_sets, target_set_color, target_set_name, dims, plotname, dirname)
	plotly()
	rs = get_subsets(reachable_sets, dims)
	ts = get_subsets(target_sets, dims)
	xlims, ylims = get_lims(reachable_sets, dims)
	xlims = expand_lims(xlims)
	ylims = expand_lims(ylims)
	p = plot(rs, color="yellow", xlim=xlims, ylim=ylims)
	plot!(ts, color=target_set_color, label=target_set_name)
	Plots.savefig(p, dirname*plotname*".html")
end

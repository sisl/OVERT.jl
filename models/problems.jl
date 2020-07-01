using LazySets
export
    OvertProblem,
    OvertQuery,
	InfiniteHyperrectangle


mutable struct OvertProblem
	true_dynamics
	overt_dynamics
	update_rule
	input_vars
	control_vars
end

mutable struct OvertQuery
	problem::OvertProblem
	network_file::String
	last_layer_activation ##::ActivationFunction
	solver::String
	ntime::Int64
	dt::Float64
	N_overt::Int64
end

Base.copy(x::OvertQuery) = OvertQuery(
	x.problem,
	x.network_file,
	x.last_layer_activation,
	x.type,
	x.ntime,
	x.dt,
	x.N_overt
	)

# this datastructure allows the hyperrectnagle to have inifinite length.
# used for satisfiability target.
struct InfiniteHyperrectangle
	low
	high
	function InfiniteHyperrectangle(low, high)
		@assert all(low .≤	high) "low must not be greater than high"
		return new(low, high)
	end
end

import LazySets.low
import LazySets.high
low(x::InfiniteHyperrectangle) = x.low
high(x::InfiniteHyperrectangle) = x.high

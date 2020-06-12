using LazySets
export
    OvertProblem,
    OvertQuery


struct OvertProblem
	true_dynamics
	overt_dynamics
	update_rule
	input_vars
	control_vars
end

struct OvertQuery
	problem::OvertProblem
	network_file::String
	last_layer_activation ##::ActivationFunction
	type::String
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

function model7_dynamics(x::Array{T, 1} where {T <: Real},
	                     u::Array{T, 1} where {T <: Real})

    dx1 = x[3]^3 - x[2]
    dx2 = x[3]
    dx3 = u[1]
    return [dx1, dx2, dx3]
end

function model7_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
	                              N_OVERT::Int,
							      t_idx::Union{Int, Nothing}=nothing)
	if isnothing(t_idx)
		v1 = :(x3^2 - x2)
	else
    	v1 = "x3_$t_idx ^3 - x2_$t_idx"
    	v1 = Meta.parse(v1)
	end
    oA_out = overapprox_nd(v1, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function model7_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => overt_output_vars[1],
                           input_vars[2] => input_vars[3],
                           input_vars[3] => control_vars[1])
    return integration_map
end

model7_input_vars = [:x1, :x2, :x3]
model7_control_vars = [:u]

Model7 = OvertProblem(
	model7_dynamics,
	model7_dynamics_overt,
	model7_update_rule,
	model7_input_vars,
	model7_control_vars
)

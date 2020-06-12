function tora_dynamics(x::Array{T, 1} where {T <: Real},
                       u::Array{T, 1} where {T <: Real})
    ϵ = 0.1
    dx1 = x[2]
    dx2 = ϵ * sin(x[3]) - x[1]
    dx3 = x[4]
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end

function tora_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                             N_OVERT::Int,
					         t_idx::Union{Int, Nothing}=nothing)
    ϵ = 0.1
	if isnothing(t_idx)
		v1 = :($ϵ * sin(x3) - x1)
	else
    	v1 = "$ϵ * sin(x3_$t_idx) - x1_$t_idx"
    	v1 = Meta.parse(v1)
	end

    oA_out = overapprox_nd(v1, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function tora_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => control_vars[1])
    return integration_map
end

tora_input_vars = [:x1, :x2, :x3, :x4]
tora_control_vars = [:u]

Tora = OvertProblem(
	tora_dynamics,
	tora_dynamics_overt,
	tora_update_rule,
	tora_input_vars,
	tora_control_vars
)

function ball_beam_dynamics(x::Array{T, 1} where {T <: Real},
	                        u::Array{T, 1} where {T <: Real})
	g = 1.0
    dx1 = x[2]
    dx2 = -g * sin(x[3]) + x[1]*x[4]^2
    dx3 = x[4]
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end

function ball_beam_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
	                              N_OVERT::Int,
							      t_idx::Union{Int, Nothing}=nothing)
	g = 1.0
	if isnothing(t_idx)
		v1 = :(x1 * x4^2 - $g * sin(x3))
	else
    	v1 = "x1_$t_idx * x2_$t_idx - $g * sin(x3_$t_idx)"
    	v1 = Meta.parse(v1)
	end
    oA_out = overapprox_nd(v1, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function ball_beam_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => control_vars[1])
    return integration_map
end

ball_beam_input_vars = [:x1, :x2, :x3, :x4]
ball_beam_control_vars = [:u]

BallnBeam = OvertProblem(
	ball_beam_dynamics,
	ball_beam_dynamics_overt,
	ball_beam_update_rule,
	ball_beam_input_vars,
	ball_beam_control_vars
)

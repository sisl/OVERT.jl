function car_dynamics(x::Array{T, 1} where {T <: Real},
	                  u::Array{T, 1} where {T <: Real})
  lr = 1.5
  lf = 1.8

  beta = atan(lr / (lr + lf) * tan(u[2]))
  dx1 = x[4] * cos(x[3] + beta)
  dx2 = x[4] * sin(x[3] + beta)
  dx3 = x[4] / lr * sin(beta)
  dx4 = u[1]
  return [dx1, dx2, dx3, dx4]
end

function car_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                            N_OVERT::Int,
							t_idx::Union{Int, Nothing}=nothing)
	lr = 1.5
    lf = 1.8

	if isnothing(t_idx)
		v1 = :(atan($(lr / (lr + lf)) * tan(u2)))
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v1_oA.output => v1_oA.output_range))

	    v2 = :($(v1_oA.output) + x3)
	    v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v2_oA.output => v2_oA.output_range))

	    v3 = :(x4 * cos($(v2_oA.output)))
	    v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

	    v4 = :(x4 * sin($(v2_oA.output)))
	    v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

	    v5 = :($(1 / lr) * x4 * sin($(v1_oA.output)))
	    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))
	else
    	v1 = "atan($(lr / (lr + lf)) * tan(u2_$t_idx))"
		v1 = Meta.parse(v1)
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v1_oA.output => v1_oA.output_range))

	    v2 = "$(v1_oA.output) + x3_$t_idx"
		v2 = Meta.parse(v2)
	    v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v2_oA.output => v2_oA.output_range))

	    v3 = "x4_$t_idx * cos($(v2_oA.output))"
		v3 = Meta.parse(v3)
	    v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

	    v4 = "x4_$t_idx * sin($(v2_oA.output))"
		v4 = Meta.parse(v4)
	    v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

	    v5 = "$(1 / lr) * x4_$t_idx * sin($(v1_oA.output))"
		v5 = Meta.parse(v5)
	    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))
	end
    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA])
    return oA_out, [v3_oA.output, v4_oA.output, v5_oA.output]
end

function car_update_rule(input_vars::Array{Symbol, 1},
                         control_vars::Array{Symbol, 1},
						 overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(input_vars[1] => overt_output_vars[1],
                           input_vars[2] => overt_output_vars[2],
                           input_vars[3] => overt_output_vars[3],
                           input_vars[4] => control_vars[1])
    return integration_map
end

car_input_vars = [:x1, :x2, :x3, :x4]
car_control_vars = [:u1, :u2]

Car = OvertProblem(
	car_dynamics,
	car_dynamics_overt,
	car_update_rule,
	car_input_vars,
	car_control_vars
)

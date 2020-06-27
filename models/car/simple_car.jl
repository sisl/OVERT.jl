"""
A function to compute dx/dt as a function of x and u
"""
function simple_car_dynamics(x::Array{T, 1} where {T <: Real},
	                         u::Array{T, 1} where {T <: Real})

  dx1 = x[4] * cos(x[3])
  dx2 = x[4] * sin(x[3])
  dx3 = u[2]
  dx4 = u[1]
  return [dx1, dx2, dx3, dx4]
end

"""
A function to constructs OVERT approximation of the simple car dynamics.
"""
function simple_car_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
		                           N_OVERT::Int,
							       t_idx::Union{Int, Nothing}=nothing)
	if isnothing(t_idx)
	    v1 = :(x4 * cos(x3))
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)

		v2 = :(x4 * sin(x3))
		v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	else
		v1 = "x4_$t_idx * cos(x3_$t_idx)"
		v1 = Meta.parse(v1)
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)

		v2 = "x4_$t_idx * sin(x3_$t_idx)"
		v2 = Meta.parse(v2)
		v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	end
    oA_out = add_overapproximate([v1_oA, v2_oA])
    return oA_out::OverApproximation, [v1_oA.output, v2_oA.output]
end

"""
Creates the mapping that is used in discrete time dynamics.
"""
function simple_car_update_rule(input_vars::Array{Symbol, 1},
		                        control_vars::Array{Symbol, 1},
								overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(input_vars[1] => overt_output_vars[1],
                           input_vars[2] => overt_output_vars[2],
                           input_vars[3] => control_vars[2],
                           input_vars[4] => control_vars[1])
    return integration_map
end

simple_car_input_vars = [:x1, :x2, :x3, :x4]
simple_car_control_vars = [:u1, :u2]

SimpleCar = OvertProblem(
	simple_car_dynamics,
	simple_car_dynamics_overt,
	simple_car_update_rule,
	simple_car_input_vars,
	simple_car_control_vars
)

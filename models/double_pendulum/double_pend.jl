function double_pend_dynamics(x::Array{T, 1} where {T <: Real},
	                          u::Array{T, 1} where {T <: Real})

    th1, th2, dth1, dth2 = x
    u1, u2 = u
    ddth1 = (16*u1 - sin(2*th1 - 2*th2)*dth1^2 - 2*sin(th1 - th2)*dth2^2 +
           2*sin(th1 - 2*th2) + 6*sin(th1) - 16*u2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2))
    ddth2 = (2*sin(th1 - th2)*dth1^2 + 16*u2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
            - sin(th1 - th2)*dth2^2 + 8*u1))/(2 - cos(th1 - th2)^2)
    dx = [dth1, dth2, ddth1, ddth2]
    return dx
end

function double_pend_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                            N_OVERT::Int,
							t_idx::Union{Int, Nothing}=nothing)
	if isnothing(t_idx)
	    v1 = :(sin(th1))
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))

	    v2 = :(sin(th2))
	    v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))

	    v3 = :(sin(th1-th2))
	    v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

	    v4 = :(cos(th1-th2))
	    v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

	    v5 = :($(v3_oA.output)*dth1^2)
	    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))

	    v6 = :($(v3_oA.output)*dth2^2)
	    v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))

	    v7 = :(sin(th1-2*th2))
	    v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))

	    v8 = :(($(v7_oA.output) - $(v6_oA.output) + 8*u1 + 3*$(v1_oA.output) -$(v4_oA.output)*(8*u2 + $(v5_oA.output)))/(2-$(v4_oA.output)^2))
	    v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)

	    v9 = :((2*$(v5_oA.output) + 16*u2 + 4*$(v2_oA.output) -$(v4_oA.output)*(8*u1 - $(v6_oA.output) + 4*$(v1_oA.output)))/(2-$(v4_oA.output)^2))
	    v9_oA = overapprox_nd(v9, range_dict; N=N_OVERT)
	else
		v1 = "sin(th1_$t_idx)"
		v1 =  Meta.parse(v1)
	    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))

		v2 = "sin(th2_$t_idx)"
		v2 = Meta.parse(v2)
	    v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))

	    v3 = "sin(th1_$t_idx - th2_$t_idx)"
		v3 = Meta.parse(v3)
	    v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range))

		v4 = "cos(th1_$t_idx - th2_$t_idx)"
	    v4 = Meta.parse(v4)
	    v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range))

		v5 = "$(v3_oA.output) * dth1_$t_idx^2"
	    v5 = Meta.parse(v5)
	    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))

		v6 = "$(v3_oA.output) * dth2_$t_idx^2"
	    v6 = Meta.parse(v6)
	    v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))

		v7 = "sin(th1_$t_idx - 2 * th2_$t_idx)"
	    v7 = Meta.parse(v7)
	    v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
	    range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))

		v8 = "($(v7_oA.output) - $(v6_oA.output) + 8*u1_$t_idx + 3*$(v1_oA.output) -$(v4_oA.output)*(8*u2_$t_idx + $(v5_oA.output)))/(2-$(v4_oA.output)^2)"
	    v8 = Meta.parse(v8)
		v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)

		v9 = "(2*$(v5_oA.output) + 16*u2_$t_idx + 4*$(v2_oA.output) -$(v4_oA.output)*(8*u1_$t_idx - $(v6_oA.output) + 4*$(v1_oA.output)))/(2-$(v4_oA.output)^2)"
		v9 = Meta.parse(v9)
	    v9_oA = overapprox_nd(v9, range_dict; N=N_OVERT)
	end
    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA, v9_oA])
    return oA_out, [v8_oA.output, v9_oA.output]
end

function double_pend_update_rule(input_vars::Array{Symbol, 1},
                                 control_vars::Array{Symbol, 1},
						         overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(input_vars[1] => input_vars[3],
                           input_vars[2] => input_vars[4],
                           input_vars[3] => overt_output_vars[1],
                           input_vars[4] => overt_output_vars[2])
    return integration_map
end

double_pend_input_vars = [:th1, :th2, :dth1, :dth2]
double_pend_control_vars = [:u1, :u2]

DoublePendulum = OvertProblem(
	double_pend_dynamics,
	double_pend_dynamics_overt,
	double_pend_update_rule,
	double_pend_input_vars,
	double_pend_control_vars
)

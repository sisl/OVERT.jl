function acc_dynamics(x::Array{T, 1} where {T <: Real},
	                  u::Array{T, 1} where {T <: Real})
	a_lead = -2.0
	mu = 0.0001

	# x1,x2,x3 are lead vehicle variables
    dx1 = x[2] # dposition
    dx2 = x[3] # dvelocity
    dx3 = -2 * x[3] + 2 * a_lead - mu * x[2]^2 # daccel 
	# x4, x5, x6 are ego vehicle variables
	# which is opposite from the notation used in the paper...
	dx4 = x[5]
	dx5 = x[6]
	dx6 = -2 * x[6] + 2 * u[1] - mu * x[5]^2
    return [dx1, dx2, dx3, dx4, dx5, dx6]
end

function acc_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
	                              N_OVERT::Int,
							      t_idx::Union{Int, Nothing}=nothing)
	a_lead = -2.0
	mu = 0.0001
	if isnothing(t_idx)
		v1 = :(-2 * x3 + 2 * $a_lead - $mu * x2^2)
		v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT, ϵ=0.5)

		v2 = :(-2 * x6 + 2 * u1 - $mu * x5^2)
		v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT, ϵ=0.5)
	else
		#v1 = "x2_$t_idx^2" # debug idea: try with simpler dynamics
		v1 = "-2 * x3_$t_idx + 2 * $a_lead - $mu * x2_$t_idx^2"
    	v1 = Meta.parse(v1) 
		v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT, ϵ=0.5)

		#v1 = "x5_$t_idx^2" # debug idea: try with simpler dynamics
		v2 = "-2 * x6_$t_idx + 2 * u1_$t_idx - $mu * x5_$t_idx^2"
		v2 = Meta.parse(v2)
		v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT, ϵ=0.5)
	end
    oA_out = add_overapproximate([v1_oA, v2_oA])
    return oA_out, [v1_oA.output, v2_oA.output]
end

function acc_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => input_vars[3],
                           input_vars[3] => overt_output_vars[1],
                           input_vars[4] => input_vars[5],
                           input_vars[5] => input_vars[6],
                           input_vars[6] => overt_output_vars[2])
    return integration_map
end

acc_input_vars = [:x1, :x2, :x3, :x4, :x5, :x6]
acc_control_vars = [:u1]

ACC = OvertProblem(
	acc_dynamics,
	acc_dynamics_overt,
	acc_update_rule,
	acc_input_vars,
	acc_control_vars
)

"""
Note: Compared this dynamics variable ordering in combination with controller acc_controller.nnet with controller outputs in 
https://github.com/souradeep-111/sherlock/blob/master/systems_with_networks/ARCH_2019/ACC/simulation_script.m
with their different variable ordering (xxvvaa) and controller outputs match to 2 decimal places in most cases indicating successful
handling of the differing variable ordering.
Closed loop simulations will not be the same as that above script uses ode45, but it matches closely for the first few steps.
"""
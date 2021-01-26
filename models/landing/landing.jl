"""
A plane lands on a runway. 
A function to compute dx/dt as a function of the system state x and the control input u.
The neural network also take T, the approximate time to landing, but this is calculated in the input layer to the network using:
(vplane - v_thresh) / accel_scale  where only vplane is a variable. 
"""
landing_accel = 1
landing_v_thresh = 66
function landing_dynamics(x::Array{T,  1} where {T <: Real}, u::Array{T, 1} where {T <: Real})
    # state variables:
    # xc is the position of the car
    # vc is the velocity of the car
    # yp is the position of the plane
    # vp is the velocity of the plane 
    dxc = x[2]
    dvc = 2*sin(.05*x[1]) # velocity of the car is constant plus wiggle
    dyp = x[4]
    dvp = u[1]*-1*landing_accel + (1-u[1])*1*landing_accel
    # u = 0 means go around, so accelerate
    # u = 1 means continue landing, so DEcelerate
    return [dxc, dvc, dyp, dvp]
end

"""
function to construct overt approximation of landing dynamics.
"""
landing_v̇c = :(2 * sin(.05 * x1))
landing_v̇p = :(u1*-1*landing_accel + (1 - u1)*1*landing_accel)
function landing_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
    N_OVERT::Int,
    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = landing_v̇c
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
        v2 = landing_v̇p
        v2_oA = overapprox_nd(vv2, range_dict; N=N_OVERT)
    else
        v1 = "2 * sin(.05 * x1_$t_idx)"
        v1 = Meta.parse(v1)
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
        v2 = "u1_$t_idx * -1 * $(landing_accel) + (1 - u1_$t_idx) * 1 * $(landing_accel)"
        v2 = Meta.parse(v2)
        v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
    end
    oA_out = add_overapproximate([v1_oA, v2_oA])
    return oA_out::OverApproximation, [v1_oA.output, v2_oA.output]
end

"""
Creates the mapping that is used in discrete time dynamics.
"""
function landing_update_rule(input_vars::Array{Symbol, 1},
    control_vars::Array{Symbol, 1},
    overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => overt_output_vars[2]
                           )
    return integration_map
end

landing_input_vars = [:x1, :x2, :x3, :x4]
landing_control_vars = [:u1]

Landing = OvertProblem(
    landing_dynamics,
    landing_dynamics_overt,
    landing_update_rule,
    landing_input_vars,
    landing_control_vars
)

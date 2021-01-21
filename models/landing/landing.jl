"""
A plane lands on a runway. 
A function to compute dx/dt as a function of the system state x and the control input u.
"""
accel = 1
function landing_dynamics(x::Array{T,  1} where {T <: Real}, u::Array{T, 1} where {T <: Real})
    # state variables:
    # xc is the position of the car
    # vc is the velocity of the car
    # yp is the position of the plane
    # vp is the velocity of the plane 
    dxc = x[2]
    dvc = 2*sin(.05*x[1]) # velocity of the car is constant plus wiggle
    dyp = x[4]
    dvp = u[1]*-1*accel + (1-u[1])*1*accel
    # u = 0 means go around, so accelerate
    # u = 1 means continue landing, so DEcelerate
    DT = u[1]*-1*accel + (1-u[1])*1*accel # as the plane accels or decels, the time to landing changes
    return [dxc, dvc, dyp, dvp, DT]
end

"""
function to construct overt approximation of landing dynamics.
"""
landing_v̇ = :(2 * sin(.05 * x1))
function landing_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
    N_OVERT::Int,
    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = landing_v̇
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
    else
        v1 = "2 * sin(.05 * x1_$t_idx)"
        v1 = Meta.parse(v1)
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
    end
    return v1_oA::OverApproximation, [v1_oA.output]
end

"""
Creates the mapping that is used in discrete time dynamics.
"""
function landing_update_rule(input_vars::Array{Symbol, 1},
    control_vars::Array{Symbol, 1},
    overt_output_vars::Array{Symbol, 1})
    integration_map = Dict()
end

landing_input_vars = [:x1, :x2, :x3, :x4, :x5]
landing_control_vars = [:u1]

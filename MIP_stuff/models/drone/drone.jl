function drone_dynamics(x::Array{T, 1} where {T <: Real},
                       u::Array{T, 1} where {T <: Real})
    Kphi = 1
    Ktheta = 1
    Kpsi = 1
    KdotPhi = 2.79e-5
    KdotTheta = 2.8e-5
    KdotPsi = 4.35e-5
    g = 9.81 # m.s^(-2)
    m = 3.3e-2 # kg
    Ix = 1.395e-5 # kg.m^2
    Iy = 1.436e-5 # kg.m^2
    Iz = 2.173e-5 # kg.m^2

    dx1 = x[4]
    dx2 = x[5]
    dx3 = x[6]
    dx4 =  x[8] * u[1] / m
    dx5 = -x[7] * u[1] / m
    dx6 = -g + u[1] / m
    dx7 = x[10]
    dx8 = x[11]
    dx9 = x[12]
    dx10 = KdotPhi / Ix * (Kphi * (u[2] - x[7]) - x[10])
    dx11 = KdotTheta / Iy * (Ktheta * (u[3] - x[8]) - x[11])
    dx12 = KdotPsi / Iz * (Kpsi * (u[4] - x[9]) - x[12])
    return [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12]
end

function drone_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                             N_OVERT::Int,
					         t_idx::Union{Int, Nothing}=nothing)

    Kphi = 1
    Ktheta = 1
    Kpsi = 1
    KdotPhi = 2.79e-5
    KdotTheta = 2.8e-5
    KdotPsi = 4.35e-5
    g = 9.81 # m.s^(-2)
    m = 3.3e-2 # kg
    Ix = 1.395e-5 # kg.m^2
    Iy = 1.436e-5 # kg.m^2
    Iz = 2.173e-5 # kg.m^2

    if isnothing(t_idx)
        v1 = :($(1/m)*x8*u1)
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v1_oA.output => v1_oA.output_range))

        v2 = :($(-1/m)*x7*u1)
        v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v2_oA.output => v2_oA.output_range))

        v3 = :($(1/m)*u1 - $g)
        v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v3_oA.output => v3_oA.output_range))

        v4 = :($(KdotPhi / Ix) * ($Kphi * (u2 - x7) - x10))
        v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v4_oA.output => v4_oA.output_range))

        v5 = :($(KdotTheta / Iy) * ($Ktheta * (u3 - x8) - x11))
        v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v5_oA.output => v5_oA.output_range))

        v6 = :($(KdotPsi / Iz) * ($Kpsi * (u4 - x9) - x12))
        v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v6_oA.output => v6_oA.output_range))
    else
        v1 = "$(1/m) * x8_$t_idx * u1_$t_idx"
        v1 = Meta.parse(v1)
        v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v1_oA.output => v1_oA.output_range))

        v2 = "$(-1/m) * x7_$t_idx * u1_$t_idx"
        v2 = Meta.parse(v2)
        v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v2_oA.output => v2_oA.output_range))

        v3 = "$(1/m) * u1_$t_idx - $g"
        v3 = Meta.parse(v3)
        v3_oA = overapprox_nd(v3, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v3_oA.output => v3_oA.output_range))

        v4 = "$(KdotPhi / Ix) * ($Kphi * (u2_$t_idx - x7_$t_idx) - x10_$t_idx)"
        v4 = Meta.parse(v4)
        v4_oA = overapprox_nd(v4, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v4_oA.output => v4_oA.output_range))

        v5 = "$(KdotTheta / Iy) * ($Ktheta * (u3_$t_idx - x8_$t_idx) - x11_$t_idx)"
        v5 = Meta.parse(v5)
        v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v5_oA.output => v5_oA.output_range))

        v6 = "$(KdotPsi / Iz) * ($Kpsi * (u4_$t_idx - x9_$t_idx) - x12_$t_idx)"
        v6 = Meta.parse(v6)
        v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
        range_dict = merge(range_dict, Dict(v6_oA.output => v6_oA.output_range))
    end

    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA])
    return oA_out, [v1_oA.output, v2_oA.output, v3_oA.output, v4_oA.output, v5_oA.output, v6_oA.output]
end

function drone_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[4],
                           input_vars[2] => input_vars[5],
                           input_vars[3] => input_vars[6],
                           input_vars[4] => overt_output_vars[1],
                           input_vars[5] => overt_output_vars[2],
                           input_vars[6] => overt_output_vars[3],
                           input_vars[7] => input_vars[10],
                           input_vars[8] => input_vars[11],
                           input_vars[9] => input_vars[12],
                           input_vars[10] => overt_output_vars[4],
                           input_vars[11] => overt_output_vars[5],
                           input_vars[12] => overt_output_vars[6],
                           )
    return integration_map
end


drone_input_vars = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :x9, :x10, :x11, :x12]
drone_control_vars = [:u1, :u2, :u3, :u4]

Drone = OvertProblem(
	drone_dynamics,
	drone_dynamics_overt,
	drone_update_rule,
	drone_input_vars,
	drone_control_vars
)

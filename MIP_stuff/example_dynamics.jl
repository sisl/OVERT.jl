"""
This models are taken from
https://github.com/souradeep-111/sherlock_2/blob/master/systems_with_networks/
"""

"""
tora example
"""

function tora_dynamics(x, u)
    dx1 = x[3]
    dx2 = 0.1*sin(x[3]) - x[1]
    dx3 = x[4]
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end

function tora_dynamics_overt(range_dict, N_overt)
    v2 = :(0.1*sin(x3) - x1)
    oA_out = overapprox_nd(v2, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function tora_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[3],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => control_vars[1])
    return integration_map
end

"""
car example
"""

function car_dynamics(x, u)
    lr = 1.5
    lf = 1.8

    beta = atan(lr / (lr + lf) * tan(u[2]))
    dx1 = x[4] * cos(x[3] + beta)
    dx2 = x[4] * sin(x[3] + beta)
    dx3 = x[4] / lr * sin(beta)
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end


function car_dynamics_overt(range_dict, N_overt)
    lr = 1.5
    lf = 1.8

    v1 = :(atan($(lr / (lr + lf)) * tan(c2)))
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

    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA])
    return oA_out, [v3_oA.output, v4_oA.output, v5_oA.output]
end

function car_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => overt_output_vars[1],
                           input_vars[2] => overt_output_vars[2],
                           input_vars[3] => overt_output_vars[3],
                           input_vars[4] => control_vars[1])
    return integration_map
end

"""
drone example
"""

function drone_dynamics(x, u)

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

function drone_dynamics_overt(range_dict, N_overt)

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

"""
ball and beam example
"""


function ball_beam_dynamics(x, u)
    dx1 = x[2]
    dx2 = 1.6*x[3]^3 - x[3] + x[1]*x[4]^2
    dx3 = x[4]
    dx4 = u[1]
    return [dx1, dx2, dx3, dx4]
end

function ball_beam_dynamics_overt(range_dict, N_overt)
    v2 = :(1.6*x3^3 - x3 + x1*x4^2)
    oA_out = overapprox_nd(v2, range_dict; N=N_OVERT)
    return oA_out, [oA_out.output]
end

function ball_beam_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[2],
                           input_vars[2] => overt_output_vars[1],
                           input_vars[3] => input_vars[4],
                           input_vars[4] => control_vars[1])
    return integration_map
end

"""
single pendulum example
"""
function single_pend_dynamics_overt(range_dict, N_overt)
    v1 = :(T + sin(th) - 0.2*dth)
    v1_oA = overapprox_nd(v1, range_dict; N=N_overt)
    return v1_oA, [v1_oA.output]
end

function single_pend_dynamics(x, u)
    dx1 = x[2]
    dx2 = u[1] + sin(x[1]) - 0.2 * x[2]
    return [dx1, dx2]
end

function single_pend_update_rule(input_vars, control_vars, overt_output_vars)
    ddth = overt_output_vars[1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

"""
double pendulum example
"""

function double_pend_dynamics(x, u)
    th1, th2, u1, u2 = x
    T1, T2 = u
    dth1 = u1
    dth2 = u2
    du1 = (16*T1 - sin(2*th1 - 2*th2)*u1^2 - 2*sin(th1 - th2)*u2^2 +
           2*sin(th1 - 2*th2) + 6*sin(th1) - 16*T2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2))
    du2 = (2*sin(th1 - th2)*u1^2 + 16*T2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
            - sin(th1 - th2)*u2^2 + 8*T1))/(2 - cos(th1 - th2)^2)
    dx = [dth1, dth2, du1, du2]
    return dx
end
#
# function double_pend_dynamics_overt(range_dict, N_overt)
#     v1 = :((16*T1 - sin(2*th1 - 2*th2)*u1^2 - 2*sin(th1 - th2)*u2^2 +
#            2*sin(th1 - 2*th2) + 6*sin(th1) - 16*T2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2)))
#     v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)
#     range_dict = merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range))
#
#     v2 = :((2*sin(th1 - th2)*u1^2 + 16*T2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
#             - sin(th1 - th2)*u2^2 + 8*T1))/(2 - cos(th1 - th2)^2))
#     v2_oA = overapprox_nd(v2, range_dict; N=N_OVERT)
#     range_dict = merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range))
#
#     oA_out = add_overapproximate([v1_oA, v2_oA])
#     return oA_out, [v1_oA.output, v2_oA.output]
# end

function double_pend_dynamics_overt(range_dict, N_overt)
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

    v5 = :($(v3_oA.output)*u1^2)
    v5_oA = overapprox_nd(v5, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range))

    v6 = :($(v3_oA.output)*u2^2)
    v6_oA = overapprox_nd(v6, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v6_oA.output=> v6_oA.output_range))

    v7 = :(sin(th1-2*th2))
    v7_oA = overapprox_nd(v7, range_dict; N=N_OVERT)
    range_dict = merge(range_dict, Dict(v7_oA.output=> v7_oA.output_range))

    v8 = :(($(v7_oA.output) - $(v6_oA.output) + 8*T1 + 3*$(v1_oA.output) -$(v4_oA.output)*(8*T2 + $(v5_oA.output)))/(2-$(v4_oA.output)^2))
    v8_oA = overapprox_nd(v8, range_dict; N=N_OVERT)

    v9 = :((2*$(v5_oA.output) + 16*T2 + 4*$(v2_oA.output) -$(v4_oA.output)*(8*T1 - $(v6_oA.output) + 4*$(v1_oA.output)))/(2-$(v4_oA.output)^2))
    v9_oA = overapprox_nd(v9, range_dict; N=N_OVERT)

    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA, v9_oA])
    return oA_out, [v8_oA.output, v9_oA.output]
end

function double_pend_update_rule(input_vars, control_vars, overt_output_vars)
    integration_map = Dict(input_vars[1] => input_vars[3],
                           input_vars[2] => input_vars[4],
                           input_vars[3] => overt_output_vars[1],
                           input_vars[4] => overt_output_vars[2])
    return integration_map
end

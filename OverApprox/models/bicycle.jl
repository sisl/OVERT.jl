"""
file to generate benchmark bicycle car model (Ex-9) overt dynamics.

  vehicle ODE Bicycle model of a vehicle with
  states
         x(1), x(2): x,y positions
         x(3): Yaw angle (ψ)
         x(4): velocity
  control inputs
         u(1): acceleration m/s^2
         u(2): steering angle of front wheel δ_f
  dx/dt
         arg = (lr + disturbance(3))/((lr + disturbance(3)) + (lf + disturbance(2))) * tan(control_input(2));
         beta = arg - arg^3/3 + arg^5/5 - arg^7/7;
         dxdt(1) = x(4) * cos(x(3) + beta);
         dxdt(2) = x(4) * sin(x(3) + beta);
         dxdt(3) = x(4)/(lr + disturbance(3)) * sin(beta);
         dxdt(4) = control_input(1)+disturbance(1);

we'll take lr = 1.5, lf = 1.8 and disturbance = zeros(1,3);
see below link for more details
    https://github.com/souradeep-111/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019
"""

# expression of dx/dt
arg = :(0.8333*tan(c2))
beta = :($arg - ($arg)^3/3 + ($arg)^5/5 - ($arg)^7/7)
dxdt1 = :(x4*cos(($beta) + x3))
dxdt2 = :(x4*sin(($beta) + x3))
dxdt3 = :(x4*sin($beta))
#dxdt4 = :(c1)
x0 = 9.5 + 5 * radius*rand(1);
y0 = -4.5 + 5 * radius*rand(1);
z0 = 2.1 + radius*rand(1);
w0 = 1.5 + radius*rand(1);
# ranges of parameters
range_dict = Dict(:x1 => [9., 10.], :x2 => [-5., -5.], :x3 => [2., 2.5],
                  :x4 => [1., 2.],  :c1 => [-2., 2.],  :c2 => [-2., 2.])

# apply overt
dxdt1_approx = overapprox_nd(dxdt1, range_dict; N=1)
dxdt2_approx = overapprox_nd(dxdt2, range_dict; N=1)
dxdt3_approx = overapprox_nd(dxdt3, range_dict; N=1)

# call overt parser
dxdt1_approx_parser = OverApproximationParser()
dxdt2_approx_parser = OverApproximationParser()
dxdt3_approx_parser = OverApproximationParser()
parse_bound(dxdt1_approx, dxdt1_approx_parser)
parse_bound(dxdt2_approx, dxdt2_approx_parser)
parse_bound(dxdt3_approx, dxdt3_approx_parser)

# delete the file, if exists. h5 can't overwrite.
out_file_name1 = "OverApprox/models/car_dxdt1.h5"
out_file_name2 = "OverApprox/models/car_dxdt2.h5"
out_file_name3 = "OverApprox/models/car_dxdt3.h5"
isfile(out_file_name1) ? rm(out_file_name1) : nothing
isfile(out_file_name2) ? rm(out_file_name2) : nothing
isfile(out_file_name3) ? rm(out_file_name3) : nothing


write_overapproximateparser(dxdt1_approx_parser, out_file_name1,
                                               [:x1, :x2, :x3, :x4],
                                               [:c1, :c2],
                                               [dxdt1_approx.output])

write_overapproximateparser(dxdt2_approx_parser, out_file_name2,
                                              [:x1, :x2, :x3, :x4],
                                              [:c1, :c2],
                                              [dxdt2_approx.output])

write_overapproximateparser(dxdt3_approx_parser, out_file_name3,
                                               [:x1, :x2, :x3, :x4],
                                               [:c1, :c2],
                                               [dxdt3_approx.output])

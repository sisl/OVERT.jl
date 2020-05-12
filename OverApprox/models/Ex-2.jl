"""
file to generate benchmark Ex-2 overt dynamics.
see below link for more details
    https://github.com/souradeep-111/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019
"""

# expression of dx/dt
# dxdt1 = x2 which does not need overt.
dxdt2 = "c*x2^2 - x1"

dxdt2_expr = Meta.parse(dxdt2)

# ranges of parameters
range_dict = Dict(:x1 => [-0.1, 1.2],  :x2 => [-0.6, 0.6], :c =>  [-2., 2.])

# apply overt
dxdt2_approx = overapprox_nd(dxdt2_expr, range_dict; N=1)

# call overt parser
dxdt2_approx_parser = OverApproximationParser()
parse_bound(dxdt2_approx, dxdt2_approx_parser)


# delete the file, if exists. h5 can't overwrite.
out_file_name2 = "OverApprox/models/Ex-2_dxdt2.h5"
isfile(out_file_name2) ? rm(out_file_name2) : nothing

write_overapproximateparser(dxdt2_approx_parser, out_file_name2,
                                               [:x1, :x2],
                                               [:c],
                                               [dxdt2_approx.output])

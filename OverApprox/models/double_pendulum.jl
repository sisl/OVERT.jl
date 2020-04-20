"""
file to generate single pendulum overt dynamics. Assuming g=1.0, m=L=0.5 and c=0.
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

# expression of acceleration with g=1.0, m=L=0.5 and c= 0.0
u1p = "(16*T1 - sin(2*th1 - 2*th2)*u1^2 - 2*sin(th1 - th2)*u2^2 +
       2*sin(th1 - 2*th2) + 6*sin(th1) - 16*T2*cos(th1 - th2))/(3 - cos(2*th1 - 2*th2))"
u2p = "(2*sin(th1 - th2)*u1^2 + 16*T2 + 4*sin(th2) - cos(th1 - th2)*(4*sin(th1)
        - sin(th1 - th2)*u2^2 + 8*T1))/(2 - cos(th1 - th2)^2)"

u1p_expr = Meta.parse(u1p)
u2p_expr = Meta.parse(u2p)

# ranges of parameters
range_dict = Dict(:th1 => [0., 1.],  :th2 => [0., 1.],
                  :u1 =>  [-1., 1.], :u2 => [-1., 1.],
                  :T1 =>  [-2., 2.], :T2 => [-2., 2.])

# apply overt
u1p_approx = overapprox_nd(u1p_expr, range_dict; N=1)
u2p_approx = overapprox_nd(u2p_expr, range_dict; N=1)

# call overt parser
u1p_approx_parser = OverApproximationParser()
u2p_approx_parser = OverApproximationParser()
parse_bound(u1p_approx, u1p_approx_parser)
parse_bound(u2p_approx, u2p_approx_parser)


# delete the file, if exists. h5 can't overwrite.
out_file_name1 = "OverApprox/models/double_pend_acceleration_1_overt.h5"
out_file_name2 = "OverApprox/models/double_pend_acceleration_2_overt.h5"
if isfile(out_file_name1)
    rm(out_file_name1)
end
if isfile(out_file_name2)
    rm(out_file_name2)
end

write_overapproximateparser(u1p_approx_parser, out_file_name1,
                                               [:th1, :th2, :u1, :u2],
                                               [:T1, :T2],
                                               [u1p_approx.output])

write_overapproximateparser(u2p_approx_parser, out_file_name2,
                                               [:th1, :th2, :u1, :u2],
                                               [:T1, :T2],
                                               [u2p_approx.output])

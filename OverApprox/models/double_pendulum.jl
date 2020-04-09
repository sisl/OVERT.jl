"""
file to generate single pendulum overt dynamics. Assuming g=m=L=1.0 and c=0.
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

include("../src/overt_to_file.jl")
include("../src/overapprox_nd_relational.jl")

u1p = "(sin(2*th1 - 2*th2)*u1^2 + 2*sin(th1 - th2)*u2^2 - 2*T1 + sin(th1 - 2*th2) - 3*sin(th1) + 2*T2*cos(th1 - th2))/(cos(2*th1 - 2*th2) - 3)"
u2p = "(2*(sin(th1 - th2)*u1^2 + T2 + sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + T1/2 + sin(th1))))/(2 - cos(th1 - th2)^2)"

u1p_expr = Meta.parse(u1p)
u2p_expr = Meta.parse(u2p)

range_dict = Dict(:th1 => [0., 1.],  :th2 => [0., 1.],
                  :u1 =>  [-1., 1.], :u2 => [-1., 1.],
                  :T1 =>  [-2., 2.], :T2 => [-2., 2.])

u1p_approx = overapprox_nd(u1p_expr, range_dict)
u2p_approx = overapprox_nd(u2p_expr, range_dict)

marabou_friendify!(u1p_approx)
marabou_friendify!(u2p_approx)


# delete the file, if exists. h5 can't overwrite.
out_file_name1 = "OverApprox/models/double_pend_acceleration_1_overt.h5"
out_file_name2 = "OverApprox/models/double_pend_acceleration_2_overt.h5"
if isfile(out_file_name1)
    rm(out_file_name1)
end
if isfile(out_file_name2)
    rm(out_file_name2)
end
bound_2_txt(u1p_approx, out_file_name1; state_vars=[:th1, :th2, :u1, :u2], control_vars=[:T1, :T2])
bound_2_txt(u2p_approx, out_file_name2; state_vars=[:th1, :th2, :u1, :u2], control_vars=[:T1, :T2])

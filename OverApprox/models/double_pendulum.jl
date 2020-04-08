include("overt_to_file.jl")

u1p = "(sin(2*th1 - 2*th2)*u1^2 + 2*sin(th1 - th2)*u2^2 - 2*T1 + sin(th1 - 2*th2) - 3*sin(th1) + 2*T2*cos(th1 - th2))/(cos(2*th1 - 2*th2) - 3)"
u2p = "(2*(sin(th1 - th2)*u1^2 + T2 + sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + T1/2 + sin(th1))))/(2 - cos(th1 - th2)^2)"

u1p_expr = Meta.parse(u1p)
u2p_expr = Meta.parse(u2p)

range_dict = Dict(:th1 => [0., 1.], :th2 => [0., 1.],
                  :u1 =>  [0., 3.],  :u2 => [0., 3.],
                  :T1 =>  [0., 5.],  :T2 => [0., 5.])

u1p_approx = overapprox_nd(u1p_expr, range_dict)
u2p_approx = overapprox_nd(u2p_expr, range_dict)

# marabou_friendify!(u1p_approx)
# marabou_friendify!(u2p_approx)
#
# bound_2_txt(u1p_approx, "u1p.h5")
# bound_2_txt(u2p_approx, "u2p.h5")

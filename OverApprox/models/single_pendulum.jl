up = "T + sin(th)"
up_expr = Meta.parse(up)

range_dict = Dict(:th => [0., 1.], :T => [-3., 3.])
up_approx = overapprox_nd(up_expr, range_dict)

marabou_friendify!(up_approx)
bound_2_txt(up_approx, "OverApprox/src/up.h5"; state_vars=[:th, :dth], control_vars=:T)

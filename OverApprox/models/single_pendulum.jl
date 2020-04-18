"""
file to generate single pendulum overt dynamics. Assuming g=m=L=1.0 and c= 0.2
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

# expression of acceleration with g=m=L=1.0 and c= 0.2
up = "T + sin(th) - 0.2*dth"
up_expr = Meta.parse(up)

# what is this for?
function single_pendulum(th, dth, T, dt)
    thnew = th + dth*dt
    ddth = T + sin(th) - 0.2*dth
    dthnew = dth + ddth*dt
    return thnew, dthnew
end

# ranges of parameters
range_dict = Dict(:th => [0., 1.], :dth => [-1., 1.], :T => [-3., 3.])

# apply overt
up_approx = overapprox_nd(up_expr, range_dict; N=1)

# call overt parser
up_approx_parser = OverApproximationParser()
parse_bound(up_approx, up_approx_parser)


# delete the file, if exists. h5 can't overwrite.
out_file_name = "OverApprox/models/single_pend_acceleration_overt.h5"
if isfile(out_file_name)
    rm(out_file_name)
end

# write to file
write_overapproximateparser(up_approx_parser, out_file_name, [:th, :dth], [:T], [up_approx.output])


# simplify nested min and max.
# marabou_friendify!(up_approx)
# write to h5 file.
#bound_2_txt(up_approx, out_file_name; state_vars=[:th, :dth], control_vars=:T)

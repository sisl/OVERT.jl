"""
file to generate single pendulum overt dynamics. Assuming g=m=L=1.0 and c= 0.2
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

using HDF5

OVERT_FOLDER = "/home/amaleki/Dropbox/stanford/Python/OverApprox/"
include(OVERT_FOLDER * "OverApprox/src/overapprox_nd_relational.jl")
include(OVERT_FOLDER * "OverApprox/src/overt_parser.jl")

function single_pendulum(file_name)
    N_overt = h5read(file_name, "overt/N")

    x_vars = [Meta.parse(x) for x in h5read(file_name, "overt/states")]
    u_vars = [Meta.parse(u) for u in h5read(file_name, "overt/controls")]
    expr = [Meta.parse(eq) for eq in h5read(file_name, "overt/eq")]

    x_bounds = h5read(file_name, "overt/bounds/states")
    u_bounds = h5read(file_name, "overt/bounds/controls")

    range_dict = Dict(x_vars[1] => x_bounds[:, 1])
    for i = 2:length(x_vars)
        range_dict[x_vars[i]] = x_bounds[:, i]
    end
    for i = 1:length(u_vars)
        range_dict[u_vars[i]] = u_bounds[:, i]
    end

    # apply overt
    expr_approx = [overapprox_nd(e, range_dict; N=N_overt) for e in expr]
    expr_approx_parser = OverApproximationParser()
    for ea in expr_approx
        tmp_parser = OverApproximationParser()
        parse_bound(ea, tmp_parser)
        expr_approx_parser = add_overapproximateparser(expr_approx_parser, tmp_parser)
    end

    # write to file
    out_vars = [ea.output for ea in expr_approx]
    write_overapproximateparser(expr_approx_parser, file_name, x_vars, u_vars, out_vars)
end

file_name = "/home/amaleki/Dropbox/stanford/Python/OverApprox/OverApprox/models/single_pendulum_overt.h5"
single_pendulum(file_name)

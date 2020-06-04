"""
file to generate single pendulum overt dynamics. Assuming g=m=L=1.0 and c= 0.2
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

using HDF5

OVERT_FOLDER = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/" #"/home/amaleki/Dropbox/stanford/Python/OverApprox/"
include(OVERT_FOLDER * "OverApprox/src/overapprox_nd_relational.jl")
include(OVERT_FOLDER * "OverApprox/src/overt_parser.jl")

function run_overt(file_name)
    N_overt = h5read(file_name, "overt/N")

    x_vars = [Meta.parse(x) for x in h5read(file_name, "overt/states")]
    u_vars = [Meta.parse(u) for u in h5read(file_name, "overt/controls")]

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
    dynamics(range_dict, N_overt, file_name, x_vars, u_vars)
end

function dynamics(range_dict, N_OVERT, file_name, x_vars, u_vars)

    # state variables are th1, th2, u1, u2
    # control variables are T1, T2
    @assert x_vars == [:th, :dth]
    @assert u_vars == [:T]
    N_VARS = 1

    v1 = :(T + sin(th) - 0.2*dth)
    v1_oA = overapprox_nd(v1, range_dict; N=N_OVERT)

    oAP = combine_them_all([v1_oA])

    rm(file_name) # .h5 can't be overwritten.
    write_overapproximateparser(oAP, file_name,
                                       x_vars,
                                       u_vars,
                                       [v1_oA.output])

    println("With N_overt = $N_OVERT")
    println(range_dict)
    println("range of outputs: ")
    println(v1_oA.output_range)
    println("# relu's = $(length(oAP.relu_list))")
    println("# max's = $(length(oAP.max_list))")


    return oAP
end

function combine_them_all(variables)
    oAP = OverApproximationParser()
    for v in variables
        oAP_tmp = OverApproximationParser()
        parse_bound(v, oAP_tmp)
        oAP = add_overapproximateparser(oAP, oAP_tmp)
    end
    return oAP
end

#file_name = "/home/amaleki/Dropbox/stanford/Python/OverApprox/
file_name = OVERT_FOLDER * "models/single_pendulum/single_pendulum_savefile.h5"
#run_overt(file_name)

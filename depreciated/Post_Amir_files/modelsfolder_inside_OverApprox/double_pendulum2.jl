"""
file to generate double pendulum overt dynamics. Assuming g=1.0, m=L=0.5 and c=0.
the output is a .h5 file with a list of all equality, min, max and inequality constraints.
"""

using HDF5

OVERT_FOLDER = "/home/amaleki/Dropbox/stanford/Python/OverApprox/"
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
    @assert Set(x_vars) == Set([:th1, :th2, :u1, :u2])
    @assert Set(u_vars) == Set([:T1, :T2])
    N_VARS = 10

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



    oAP = combine_them_all([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA, v9_oA])

    rm(file_name)
    write_overapproximateparser(oAP, file_name,
                                       x_vars,
                                       u_vars,
                                       [v8_oA.output, v9_oA.output])

    println("With N_overt = $N_OVERT")
    println(range_dict)
    println("range of outputs: ")
    println(v8_oA.output_range)
    println(v9_oA.output_range)
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


file_name = "/home/amaleki/Dropbox/stanford/Python/OverApprox/OverApprox/models/double_pendulum2_savefile.h5"
out = run_overt(file_name);

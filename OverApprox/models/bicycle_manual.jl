include("../src/overapprox_nd_relational.jl")
include("../src/overt_parser.jl")

N_VARS = 10

range_dict = Dict(:c2=>[0.,1.5], :x3=> [1., 3.], :x4 => [1., 3.])

v1 = :(0.833*tan(c2))
v1_oA = overapprox_nd(v1, range_dict; N=1)

#v2 = :($(v1_oA.output) - $(v1_oA.output)^3/3 + $(v1_oA.output)^5/5 - $(v1_oA.output)^7/7)
v2 = :(sin($(v1_oA.output)))
v2_oA = overapprox_nd(v2, merge(range_dict, Dict(v1_oA.output=> v1_oA.output_range)); N=1)

v3 = :($(v2_oA.output) + x3)
v3_oA = overapprox_nd(v3, merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range)); N=1)

v4 = :(cos($(v3_oA.output)))
v4_oA = overapprox_nd(v4, merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range)); N=1)

v5 = :(sin($(v3_oA.output)))
v5_oA = overapprox_nd(v5, merge(range_dict, Dict(v3_oA.output=> v3_oA.output_range)); N=1)

v6 = :(x4*$(v4_oA.output))
v6_oA = overapprox_nd(v6, merge(range_dict, Dict(v4_oA.output=> v4_oA.output_range)); N=1)

v7 = :(x4*$(v5_oA.output))
v7_oA = overapprox_nd(v7, merge(range_dict, Dict(v5_oA.output=> v5_oA.output_range)); N=1)

v8 = :(0.667*sin($(v2_oA.output)))
v8_oA = overapprox_nd(v8, merge(range_dict, Dict(v2_oA.output=> v2_oA.output_range)); N=1)

function combine_them_all()
    oAP = OverApproximationParser()
    for v in [v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA, v7_oA, v8_oA]
        oAP_tmp = OverApproximationParser()
        parse_bound(v, oAP_tmp)
        oAP = add_overapproximateparser(oAP, oAP_tmp)
    end
    return oAP
end

oAP = combine_them_all()


# delete the file, if exists. h5 can't overwrite.
out_file_name = "OverApprox/models/car_dxdt.h5"
isfile(out_file_name) ? rm(out_file_name) : nothing

write_overapproximateparser(oAP, out_file_name,
                               [:x1, :x2, :x3, :x4],
                               [:c1, :c2],
                               [v6_oA.output, v7_oA.output, v8_oA.output])

# testing script for overt_parser.jl

include("OverApprox/src/overapprox_nd_relational.jl")
include("OverApprox/src/overt_parser.jl")

oa = overapprox_nd(:(sin(x + y)/y), Dict(:x=>[2,3], :y=>[1,2]))

oAP = OverApproximationParser()

parse_bound(oa, oAP)

print_overapproximateparser(oAP)

state_vars = [:x]
control_vars = [:y]
output_vars = [:x]
write_overapproximateparser(oAP::OverApproximationParser,
                            "bound_test.hdf5"::String,
                            state_vars,
                            control_vars,
                            output_vars)

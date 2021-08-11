# run all tests

using NBInclude

include("d2f_zeros_test.jl")
include("overapprox_nd_unittest.jl")

@testset "test methods notebook" begin
    @nbinclude(joinpath(dirname(@__DIR__), "plots", "Methods Section.ipynb"))
end
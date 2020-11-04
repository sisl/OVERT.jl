# test d2f zeros lookup functions

include("../src/overt_utils.jl")

# test get_regions_1arg
# division
# c/x, c > 0, c<0, interval all pos or all neg
@assert get_regions_1arg(:(5/x), :x, 1, 2) == ([], true)
@assert get_regions_1arg(:(5/x), :x, -2, -1) == ([], false)
@assert get_regions_1arg(:(-5/x), :x, 1, 2) == ([], false)
@assert get_regions_1arg(:(-5/x), :x, -2, -1) == ([], true)
# exponent
# c^x
@assert get_regions_1arg(:(5.1^x), :x, -1, 1) == ([], true)
# x^c
# fractional exponent
@assert get_regions_1arg(:(x^5.1), :x, 0, 2) == ([], true)
# odd exponent
@assert get_regions_1arg(:(x^5), :x, 1, 2) == ([], true)
@assert get_regions_1arg(:(x^5), :x, -2, -1) == ([], false)
@assert get_regions_1arg(:(x^5), :x, -1, 2) == ([0], nothing)
# even exponent
@assert get_regions_1arg(:(x^6), :x, -2, -1) == ([], true)

# test get_regions_unary
# sin, at boundary points
@assert all(get_regions_unary(:sin, -π, π)[1] .≈ [-π, 0, π])
@assert get_regions_unary(:sin, -π, π)[2] == nothing
# at fractional points
@assert get_regions_unary(:sin, -1, 1) == ([0], nothing)
@assert all(get_regions_unary(:sin, 1, 4)[1] .≈ [π])
@assert get_regions_unary(:sin, 1, 4)[2] == nothing
# cos
@assert get_regions_unary(:cos, -1, 1) == ([], false)
@assert all(get_regions_unary(:cos, -π, π)[1] .≈ [-π/2, π/2])
@assert get_regions_unary(:cos, -π, π)[2] == nothing
# exp
@assert get_regions_unary(:exp, -1, 1) == ([], true)
# log
@assert get_regions_unary(:log, .5, 1.5) == ([], false)
# tanh
@assert get_regions_unary(:tanh, -2, 2) == ([0], nothing)
@assert get_regions_unary(:tanh, -2, -1) == ([], true)
@assert get_regions_unary(:tanh, 1, 2) == ([], false)

println("All tests pass!")
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

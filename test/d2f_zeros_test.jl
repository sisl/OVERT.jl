# test d2f zeros lookup functions

include("../src/overt_utils.jl")
using Test

@testset "Testing Second Derivative Zero Computation" begin
    # test get_regions_1arg
    # division
    # c/x, c > 0, c<0, interval all pos or all neg
    @test get_regions_1arg(:(5/x), :x, 1, 2) == ([], true)
    @test get_regions_1arg(:(5/x), :x, -2, -1) == ([], false)
    @test get_regions_1arg(:(-5/x), :x, 1, 2) == ([], false)
    @test get_regions_1arg(:(-5/x), :x, -2, -1) == ([], true)
    # exponent
    # c^x
    @test get_regions_1arg(:(5.1^x), :x, -1, 1) == ([], true)
    # x^c
    # fractional exponent
    @test get_regions_1arg(:(x^5.1), :x, 0, 2) == ([], true)
    # odd exponent
    @test get_regions_1arg(:(x^5), :x, 1, 2) == ([], true)
    @test get_regions_1arg(:(x^5), :x, -2, -1) == ([], false)
    @test get_regions_1arg(:(x^5), :x, -1, 2) == ([0], nothing)
    # even exponent
    @test get_regions_1arg(:(x^6), :x, -2, -1) == ([], true)

    # test get_regions_unary
    # sin, at boundary points
    @test all(get_regions_unary(:sin, -π, π)[1] .≈ [-π, 0, π])
    @test get_regions_unary(:sin, -π, π)[2] === nothing
    # at fractional points
    @test get_regions_unary(:sin, -1, 1) == ([0], nothing)
    @test all(get_regions_unary(:sin, 1, 4)[1] .≈ [π])
    @test get_regions_unary(:sin, 1, 4)[2] === nothing
    # cos
    @test get_regions_unary(:cos, -1, 1) == ([], false)
    @test all(get_regions_unary(:cos, -π, π)[1] .≈ [-π/2, π/2])
    @test get_regions_unary(:cos, -π, π)[2] === nothing
    # exp
    @test get_regions_unary(:exp, -1, 1) == ([], true)
    # log
    @test get_regions_unary(:log, .5, 1.5) == ([], false)
    # tanh
    @test get_regions_unary(:tanh, -2, 2) == ([0], nothing)
    @test get_regions_unary(:tanh, -2, -1) == ([], true)
    @test get_regions_unary(:tanh, 1, 2) == ([], false)
end
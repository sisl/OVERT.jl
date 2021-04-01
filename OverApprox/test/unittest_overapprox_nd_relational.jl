using Debugger
using Revise

N_VARS = 0

@assert add_var(OverApproximation()) == :v_1

@assert apply_fx(:(x + y), :a) == :(a + y)

@assert is_number(:(log(2)/3))

@assert is_unary(:(sin(x)))
@assert !is_unary(:(x + y))

@assert is_binary(:(x + y))

function affine_tests()
    @assert is_affine(:(x+1))
    @assert is_affine(:(-x+(2y-z)))
    @assert !is_affine(:(log(x)))
    @assert !is_affine(:(x + x*z))
    @assert is_affine(:(x/6))
    @assert is_affine(:(5*x))
    @assert is_affine(:((1 / 6) * x))
    @assert is_affine(:(log(2)*x))
    @assert is_affine(:(-x))
end
affine_tests()

# function outer_affine_tests()
#     @assert is_outer_affine(:(-sin(x) + sin(y)))
#     @assert is_outer_affine(:( -5*(sin(x) - 3*sin(y)) ) )
# end
# outer_affine_tests()

overapprox_nd(:(sin(x)), Dict(:x=>[0,π/2]))

overapprox_nd(:(sin(sin(x))), Dict(:x=>[0,π/2]))

# improve how this is handled
overapprox_nd(:(sin(sin(z + y))), Dict(:z=>[0,π], :y=>[-π,π]))

overapprox_nd(:(2*x), Dict(:x=>[0,1]))

# TODO: This isn't reduced to eval(log(2))*x because whole thing is affine...
overapprox_nd(:(log(2)*x), Dict(:x=>[0,1]))

overapprox_nd(:(2*log(x)), Dict(:x=>[1,2]))

expr = :(exp(2*sin(x) + y) - log(6)*z)
init_set = Dict{Symbol, Array{Float64,1}}(:x=>[0,1], :z=>[0,π], :y=>[-π,π])
oa = overapprox_nd(expr, init_set)

#
b1 = OverApproximation()
b1.ranges = Dict(:A=>[1.,2.], :B=>[3.,4.])
b1.nvars = 2
expand_multiplication(:A, :B, b1; ξ=0.1)
# reference: (:(exp(log(C) + log(D)) + (1.0 - 0.1) * D + (3.0 - 0.1) * C + (1.0 - 0.1) * (3.0 - 0.1)), Dict(:A => (1.0, 2.0),:D => (0.1, 1.1),:B => (3.0, 4.0),:C => (0.1, 1.1)), Expr[:(C = (A - 1.0) + 0.1), :(D = (B - 3.0) + 0.1)])

bound_binary_functions(:*, :A, :B, b1)

xxx = 0; yyy = 1; zzz = 2; kkk = -10;
e1 = :(xxx + yyy + zzz + kkk - xxx - yyy - zzz - kkk)
e2 = reduce_args_to_2(e1)
eval(e1) == eval(e2)

overapprox_nd(:(x*y), Dict(:x=>[1,2], :y=>[-10,-9]))

overapprox_nd(:(sin(6) + sin(x)), Dict(:x=>[1,2]))

overapprox_nd(:(sin(6)*sin(x)), Dict(:x=>[1,2]))

overapprox_nd(:(sin(6)*sin(x)*sin(y)), Dict(:x=>[1,2], :y=>[1,2])::Dict{Symbol,Array{Int64,1}})

# TODO: when the whole expression is affine, division by a constant still appears...fix so it's converted to multiplication by 1/c
overapprox_nd(:(x/6), Dict(:x=>[-1,1]))
# should be handled by affine base case

overapprox_nd(:(6/x), Dict(:x=>[1,2]))
# should be handled by 1d base case

overapprox_nd(:(6/(x+y)), Dict(:x=>[2,3], :y=>[-1,2]))

overapprox_nd(:(x/y), Dict(:x=>[2,3], :y=>[1,2]))

overapprox_nd(:(sin(x + y)/6), Dict(:x=>[2,3], :y=>[1,2]))

overapprox_nd(:(sin(x + y)/y), Dict(:x=>[2,3], :y=>[1,2]))

overapprox_nd(:(exp(x^2)), Dict(:x=>[-1,1]))

overapprox_nd(:((x+sin(y))^3), Dict(:x=>[2,3], :y=>[1,2]))

overapprox_nd(:(2^x), Dict(:x=>[2,3]))

overapprox_nd(:(-sin(x+y)), Dict(:x=>[2,3], :y=>[1,2]))

overapprox_nd(:(log(x)), Dict(:x => [1.0, 166.99205596346707]); N=-1)



# todo:
# find good way to visualize overapprox and/or qualitatively validate overapprox
# quantitative validation: dreal
# analytical (symbolic) differentiation in overest_new.jl

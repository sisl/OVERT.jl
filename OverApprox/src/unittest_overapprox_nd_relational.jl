include("OverApprox/src/overapprox_nd_relational.jl")
using Debugger
using Revise

@assert add_var(OverApproximation()) == :vA

@assert apply_fx(:(x + y), :a) == :(a + y)

@assert is_number(:(log(2)/3))

@assert is_unary(:(sin(x)))
@assert !is_unary(:(x + y))

@assert is_binary(:(x + y))

overapprox_nd(:(sin(x)), Dict(:x=>[0,π/2]))

overapprox_nd(:(sin(sin(x))), Dict(:x=>[0,π/2]))

overapprox_nd(:(sin(sin(z + y))), Dict(:z=>[0,π], :y=>[-π,π]))

overapprox_nd(:(2*x), Dict(:x=>[0,1]))

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

function test_reduce_args_to_2()
    x = 0; y = 1; z = 2; k = -10;
    e1 = :(x + y + z + k - x - y - z - k)
    e2 = reduce_args_to_2(e1)
    eval(e1) == eval(e2)
end
@assert test_reduce_args_to_2()

overapprox_nd(:(x*y), Dict(:x=>[1,2], :y=>[-10,-9]))

overapprox_nd(:(sin(6) + sin(x)), Dict(:x=>[1,2]))

overapprox_nd(:(sin(6)*sin(x)), Dict(:x=>[1,2]))

overapprox_nd(:(sin(6)*sin(x)*sin(y)), Dict(:x=>[1,2], :y=>[1,2])::Dict{Symbol,Array{Int64,1}})

overapprox_nd(:(exp(x^2)), Dict(:x=>[-1,1]))

# todo:
# find good way to visualize overapprox and/or qualitatively validate overapprox
# quantitative validation: dreal
# analytical (symbolic) differentiation in overest_new.jl
# handle division by scalars (multiplication really of 1/the_scalar...)
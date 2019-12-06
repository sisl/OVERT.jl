include("OverApprox/src/overapprox_nd_relational.jl")
using Debugger

@assert get_new_var(0) == (:A,1)

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

overapprox_nd(:(exp(2*sin(x) + y) - log(6)*z), Dict(:x=>[0,1], :z=>[0,π], :y=>[-π,π]) )


# todo:
# find good way to visualize overapprox and/or qualitatively validate overapprox
# bug highlighted above
# parsing of multiplication
# analytical (symbolic) differentiation in overest_new.jl
# handle division by scalars (multiplication really of 1/the_scalar...)
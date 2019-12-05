include("OverApprox/src/overapprox_nd_relational.jl")

@assert get_new_var(0) == (:A,1)

@assert apply_fx(:(x + y), :a) == :(a + y)

@assert is_number(:(log(2)/3))

@assert is_unary(:(sin(x)))
@assert !is_unary(:(x + y))

@assert is_binary(:(x + y))

overapprox_nd(:(sin(x)), Dict(:x=>[0,π/2]))
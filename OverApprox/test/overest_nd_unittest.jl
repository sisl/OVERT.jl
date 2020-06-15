include("overest_nd.jl")

# tests for overest_nd.jl

# test for find_variables
@assert find_variables(:(x+1)) == [:x]
@assert find_variables(:(log(x + y*z))) == [:x, :y, :z]

# test for is_affine
@assert is_affine(:(x+1)) == true
@assert is_affine(:(-x+(2y-z))) == true
@assert is_affine(:(log(x))) == false
@assert is_affine(:(x+ xz)) == true # interprets xz as one variable
@assert is_affine(:(x+ x*z)) == false
@assert is_affine(:(x + y*log(2))) == true

# test for is_1d
@assert is_1d(:(x*log(2))) == true
@assert is_1d(:(x^2.5)) == true

# test for is_unary
@assert is_unary(:(x+y)) == false
@assert is_unary(:(x^2.5)) == false # this is like pow(x, 2.5), hence not unary. is taken care of in is_1d
@assert is_unary(:(sin(x))) == true
@assert is_unary(:(x*y)) == false

#TODO: investigate this one.
#


# test find_UB (really tests for overest_new.jl)

# test find_affine_range
@assert find_affine_range(:(x + 1), Dict(:x => (0, 1))) == (1,2)
@assert find_affine_range(:(- x + 1), Dict(:x => (0, 1))) == (0,1)
@assert find_affine_range(:((-1 + y) + x), Dict(:x => (0,1), :y => (1,2))) == (0,2)
@assert find_affine_range(:(2*x), Dict(:x => (0,1))) == (0,2)
@assert find_affine_range(:(x*2), Dict(:x => (0,1))) == (0,2)

# test substitute
@assert substitute!(:(x^2+1), :x, :(y+1)) == :((y+1)^2+1)

# test count min max
@assert count_min_max(:(min(x) + max(y - min(x)))) == [2,1]

# test upperbound_expr_compositions
# need clarity on precise purpose and signature of the function

# test reduce_args_to_2!
@assert reduce_args_to_2(:(x+y+z)) == :(x+(y+z))
@assert reduce_args_to_2(:(sin(x*y*z))) == :(sin(x*(y*z)))
# bug ^ doesn't seem to modify sin(x*y*z)

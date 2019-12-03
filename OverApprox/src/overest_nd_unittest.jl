include("overest_nd.jl")

# tests for overest_nd.jl

# test for find_variables
@assert find_variables(:(x+1)) == [:x]
@assert find_variables(:(log(x + y*z))) == [:x, :y, :z]

# test for is_affine
@assert is_affine(:(x+1)) == true
@assert is_affine(:(-x+(2y-z))) == true 
# tests handling of unary operators like (- x)
@assert is_affine(:(log(x))) == false
@assert is_affine(:(x+ xz)) == true # interprets xz as one variable
@assert is_affine(:(x+ x*z)) == false
# bookmark: putting in print statements to debug

# test find_UB (really tests for overest_new.jl)

# test get_range

# test substitute

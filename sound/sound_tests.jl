# put testing code here for all code in the sound subdirectory
# make sure to call the test code in this script

include("../OverApprox/src/overapprox_nd_relational.jl")
include("soundness.jl")

# low level testing
f1 = FormulaStats()
get_new_macro(f1)

declare_list([:(x <= 5), :(y>=6)], f1)

declare_list([:(x <= max(5,y)), :(y>= 6 * relu(x-y+log(6)))], f1)

f = assert_negation_of_conjunction([:(x <= max(5,y)), :(y>= 6 * relu(x-y+log(6)))], f1; conjunct_name="phihat")
print(join(f,"\n"))

check_soundness("dummy_sin")


# high level testing
# def linear_plant_test():
#     """
#     linear plant test
#     # phi
#     y = x  
#     # phi hat
#     OA: x - 1 <= y <= x + 1
#     """
#     print("~~~~~~~~~~~lin plant test~~~~~~~~~~~~~~~\n")
#     # phi
#     x = 'x'
#     y = 'y'
#     phi = [Constraint("EQUALITY", [Monomial(1, x), Monomial(-1,y)], 0)] # x - y == 0
#     # phi hat
#     c1 = Constraint("LESS_EQ", [Monomial(1, x), Monomial(-1,y)], 1 ) # x - y <= 1
#     c2 = Constraint("LESS_EQ", [Monomial(-1, x), Monomial(1,y)], 1 )# y - x <= 1
#     phi_hat = [c1, c2]
#     # check!
#     oav = OverapproxVerifier(phi, phi_hat)
#     oav.create_smtlib_script() # should write to file that can be checked with: https://cvc4.github.io/app/
#     # copy and paste smtlib2 formula into https://cvc4.github.io/app/
#     # should be unsat

# linear_plant_test()


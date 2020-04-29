import numpy as np
from check_overapprox import FormulaConverter, OverapproxVerifier
from MC_constraints import Constraint, Monomial, MaxConstraint, ReluConstraint, MatrixConstraint

f = FormulaConverter()

print(f.prefix_notate("+", ["A", "B"]))

print(f.declare_const("A", "Bool"))

print(f.define_atom("A", "(< y 5)"))

print(f.negate("(< y 5)"))

c1 = Constraint('LESS_EQ', [Monomial(-6, "x"), Monomial(5, "y")], -2)
print(f.convert_Constraint(c1))

c2 = MaxConstraint(['v1','v2'], 'v3')
print(f.convert_MaxConstraint(c2))

c3 = ReluConstraint('p', 'q')
print(f.convert_ReluConstraint(c3))

c4 = MatrixConstraint('EQUALITY', A=np.random.rand(2,2), x=np.array([['x'],['y']], dtype='object'), b=np.zeros((2,1)) )
print(f.convert_MatrixConstraint(c4))

print(f.declare_list([c1, c1, c2, c2, c3, c3, c4]))

print('\n'.join(f.declare_conjunction([c1, c2, c3, c4])[0]))

def linear_plant_test():
    """
    linear plant test
    # phi
    y = x  
    # phi hat
    OA: x - 1 <= y <= x + 1
    """
    print("~~~~~~~~~~~lin plant test~~~~~~~~~~~~~~~\n")
    # phi
    x = 'x'
    y = 'y'
    phi = [Constraint("EQUALITY", [Monomial(1, x), Monomial(-1,y)], 0)] # x - y == 0
    # phi hat
    c1 = Constraint("LESS_EQ", [Monomial(1, x), Monomial(-1,y)], 1 ) # x - y <= 1
    c2 = Constraint("LESS_EQ", [Monomial(-1, x), Monomial(1,y)], 1 )# y - x <= 1
    phi_hat = [c1, c2]
    # check!
    oav = OverapproxVerifier(phi, phi_hat)
    oav.create_smtlib_script() # should write to file that can be checked with: https://cvc4.github.io/app/
    # copy and paste smtlib2 formula into https://cvc4.github.io/app/
    # should be unsat

def nonlinear_plant_test():
    pass

linear_plant_test()
nonlinear_plant_test()
# unit test for properties

from properties import ConstraintProperty
from MC_constraints import Constraint, ConstraintType, Monomial

# example 1
def ex1():
    print("~~~~~ test 1 ~~~~~")
    m = [Monomial(5,'x'), Monomial(6, 'y')]
    c = Constraint('LESS', monomials=m, scalar=5)
    # 5x 6y < 5
    print(c)
    print(c.complement())
    #
    p = ConstraintProperty([c])
    print(p)
    print(p.complement())

# example 2
def ex2():
    print("~~~~~ test 2 ~~~~~")
    m1 = [Monomial(5,'x'), Monomial(6, 'y')]
    c1 = Constraint('LESS', monomials=m1, scalar=5)
    # 5x 6y < 5
    print(c1)
    print(c1.complement())
    #
    m2 = [Monomial(4,'x'), Monomial(5, 'y')]
    c2 = Constraint('LESS', monomials=m2, scalar=4)
    # 4x 5y < 4
    print(c2)
    print(c2.complement())
    #
    p = ConstraintProperty([c1,c2])
    print(p)
    print(p.complement())

# example 3
def ex3():
    print("~~~~~ test 3 ~~~~~")
    m = [Monomial(-5,'x'), Monomial(-6, 'y')]
    c = Constraint('GREATER', monomials=m, scalar=-5)
    # -5x -6y < -5
    print(c)
    print(c.complement())
    #
    p = ConstraintProperty([c])
    print(p)
    print(p.complement())

# example 4
def ex4():
    print("~~~~~ test 4 ~~~~~")
    m1 = [Monomial(-5,'x'), Monomial(-6, 'y')]
    c1 = Constraint('GREATER', monomials=m1, scalar=-5)
    # -5x -6y < -5
    #
    m2 = [Monomial(-4,'x'), Monomial(-5, 'y')]
    c2 = Constraint('GREATER', monomials=m2, scalar=-4)
    # -4x -5y < -4
    #
    p = ConstraintProperty([c1,c2])
    print(p)
    print(p.complement())

# example 4
def ex5():
    print("~~~~~ test 5 ~~~~~")
    m1 = [Monomial(-5,'x'), Monomial(-6, 'y')]
    c1 = Constraint('GREATER', monomials=m1, scalar=-5)
    # -5x -6y < -5
    #
    m2 = [Monomial(-4,'x'), Monomial(-5, 'y')]
    c2 = Constraint('GREATER', monomials=m2, scalar=-4)
    # -4x -5y < -4
    m3 = [Monomial(-3,'x'), Monomial(-4, 'y')]
    c3 = Constraint('GREATER', monomials=m3, scalar=-3)
    # -3x -4y < -3
    #
    p = ConstraintProperty([c1,c2,c3])
    print(p)
    print(p.complement())



ex1()
ex2()
ex3()
ex4()
ex5()
print("all tests pass!")



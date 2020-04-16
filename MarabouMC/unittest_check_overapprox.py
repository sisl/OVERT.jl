import numpy as np
from check_overapprox import FormulaConverter
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


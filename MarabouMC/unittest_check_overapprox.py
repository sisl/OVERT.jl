from check_overapprox import FormulaConverter
from MC_constraints import Constraint, Monomial

f = FormulaConverter()

print(f.prefix_notate("+", ["A", "B"]))

print(f.declare_const("A", "Bool"))

print(f.define_atom("A", "(< y 5)"))

print(f.negate("(< y 5)"))

c = Constraint('LESS_EQ', [Monomial(-6, "x"), Monomial(5, "y")], -2)
print(f.convert_Constraint(c))

print(f.declare_list([c, c]))


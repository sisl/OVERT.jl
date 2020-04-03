import h5py
from MarabouMC.MC_constraints import *

import h5py
f = h5py.File("OverApprox/src/u1p.h5", "r")
n_eq = len(f['/eq/']) // 3
n_min = len(f['/min/']) // 2
n_max = len(f['/max/']) // 2

eq_list = []
for i in range(n_eq):
    v = f['/eq/v%d'%(i+1)][()]
    c = f['/eq/c%d'%(i+1)][()]
    b = f['/eq/b%d'%(i+1)][()]
    monomial_list = [Monomial(cs, vs) for (cs, vs) in zip(c, v)]
    eq_list.append(Constraint(ConstraintType('EQUALITY'), monomial_list, b))

min_list = []
for i in range(n_min):
    v = f['/min/v%d'%(i+1)][()]
    c = f['/min/c%d'%(i+1)][()]
    monomial_list = [Monomial(c[0], v[1]), Monomial(c[1], v[2])]
    min_list.append(MinConstraint(monomial_list, v[0]))

relu_list = []
for i in range(n_max):
    v = f['/max/v%d'%(i+1)][()]
    c = f['/max/c%d'%(i+1)][()]
    assert('0' in v)
    vin = v[2]
    if vin == "0":
        vin = v[1]
    relu_list.append(ReluConstraint(vin, v[0]))
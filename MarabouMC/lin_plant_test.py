## Testing a linear plant model in the model checker

import numpy as np
from MC_constraints import Constraint, ConstraintType, Monomial
from MC_interface import BMC, TransitionRelation, ConstraintProperty
from marabou_interface import MarabouWrapper

tr = TransitionRelation()
# States
tr.states = ["x", "y"]
tr.next_states = [s+"'" for s in tr.states]
# Constraints
# x' = x + y   ->   x + y - x' = 0
c1 = Constraint(ConstrantType('EQUALITY'))
c1.monomials = [Monomial(1, "x"), Monomial(1,"y"), Monomial(-1,"x'")]
# y' = y  ->  y - y' = 0
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1,"y"), Monomial(-1, "y'")]
tr.constraints = [c1, c2]
# initial set
init_set = {"x": (0.1,1), "y": (-1,1)}
# build the transition system as a (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

# property
p = Constraint(ConstraintType('GREATER'))
# x > 0 (complement will be x <= 0)
p.monomials = [Monomial(1, "x")]
prop = ConstraintProperty([p])

# algo
algo = BMC(ts = transition_system, prop = ConstraintProperty, solver=solver)
algo.check_invariant_until(3)





import numpy as np
from keras.models import load_model
from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from MC_simulate import simulate_Ex_2

# This is controller that was used for this benchmark.
controller_file_name = "/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_2/modified_controller_keras.h5"
model = load_model(controller_file_name)
controller = KerasController(keras_model=model)

# create overt dynamics objects
overt_obj = OvertConstraint("../OverApprox/models/Ex-2_dxdt2.h5")

# setup states, control and dynamics variables.
states = overt_obj.state_vars
x1 = states[0]
x2 = states[1]

controls = overt_obj.control_vars
dxdt2 = overt_obj.output_vars[0]

ex_2_dynamics = Dynamics(None, np.array(states).reshape(-1, 1), np.array(controls).reshape(-1, 1))
next_states = ex_2_dynamics.next_states.reshape(2,)
next_x1 = next_states[0]
next_x2 = next_states[1]

dt = 0.01

# x1_next = x1 + dx1dt*dt, dx1 = x2
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, x1), Monomial(dt, x2), Monomial(-1, next_x1)]
print(c1.monomials)

# u1_next = u1 + dx2dt*dt, dx2 = x2^2*c - x1
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1, x2), Monomial(dt, dxdt2), Monomial(-1, next_x2)]
print(c2.monomials)

dynamics_constraints = [c1, c2]
dynamics_constraints += overt_obj.constraints
ex_2_dynamics.constraints = dynamics_constraints

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=ex_2_dynamics,
                                        controller_obj=controller)

# initial set
init_set = {x1: (0.7, 0.9), x2: (0.4, 0.6)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

print(len([c for c in ex_2_dynamics.constraints if isinstance(c, ReluConstraint)]))
print(len([c for c in controller.constraints if isinstance(c, ReluConstraint)]))

# solver
solver = MarabouWrapper()

def constraint_variable_to_interval(variable, LB, UB):
    p1 = Constraint(ConstraintType('GREATER'))
    p1.monomials = [Monomial(1, variable)]
    p1.scalar = LB # 0 #
    #
    p2 = Constraint(ConstraintType('LESS'))
    p2.monomials = [Monomial(1, variable)]
    p2.scalar = UB
    return [p1, p2]


prop_list = []
prop_list += constraint_variable_to_interval(x1, .2, 1.) # 0.2 < x1 < 1.0
prop_list+=constraint_variable_to_interval(x2, .2, 1.) # 0.2 < x2 < 1.0
prop = ConstraintProperty(prop_list)

# algo
ncheck_invariant = 15
algo = BMC(ts=ts, prop=prop, solver=solver)
result, value, stats = algo.check_invariant_until(ncheck_invariant)

# random runs to give intuition to MC result
n_simulation = 10000
print("Now running %d simulations: " %n_simulation, end="")


# simulate failure if applicable

# things for demo:
# slide of double pendulum EOM
# show overt working
# simulate (bad) controller in the region that we test
# come to this script, run, it produces SAT and simulate(?) or plot(?) the SAT failure 

# notes just for marabou team: marabou gets slow with good controller
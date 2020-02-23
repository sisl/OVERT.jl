# script using MC
from MC_interface import *
from dynamical_systems.dynamical_systems import f1
import numpy as np

controller_file = "mystring.pb"
property_file = "my_prop1.txt"
# dynamics in python function f1

my_dynamics = OVERTDynamics(f1, ["x", "theta"], ["u"])
init_set = {"x": (0, 5), "theta": (-np.pi/4, np.pi/4), "u": (-10,10)}
tr = TFControlledOVERTTransitionRelation(dynamics_obj=my_dynamics, controller_file=controller_file)
transition_system = MyTransitionSystem(my_dynamics.states, init_set, tr)

solver = MarabouWrapper()

algo = BMC(ts = transition_system, prop_file = property_file, solver=solver)
algo.check_until(3)



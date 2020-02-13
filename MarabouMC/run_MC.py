# script using MC
from MC_interface import *

controller_file = "mystring.pb"
dynamics_file = "mystring2.txt"
property_file = "mystring3.txt"

tr = TFControlledTransitionRelation(controller=controller_file,dynamics=dynamics_file)
transition_sys = MyTransitionSystem(transition_relation=tr)

solver = MarabouWrapper()

algo = BMC(ts = transition_sys, prop_file = property_file, solver=solver)
algo.check_until(3)



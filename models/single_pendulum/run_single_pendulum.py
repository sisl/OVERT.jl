import os
import sys
assert len(sys.argv) == 3, "you should pass marabou address AND number of cores used in the job"
MARABOU_PATH = sys.argv[1]
N_CORES = int(sys.argv[2])
print(os.getcwd())
sys.path.insert(0, "../../MarabouMC")
sys.path.insert(0, MARABOU_PATH)

from example import OvertMCExample

example_1 = OvertMCExample(
             keras_controller_file="single_pend_nn_controller_ilqr_data.h5",
             overt_dynamics_file="single_pendulum.jl",
             controller_bounding_values=[[-2., 2.]],
             integration_map=['s1', 'o0'],
             model_states=[b'th', b'dth'],
             model_controls=[b'T'],
             init_range=[[-0.1, 0.1], [-0.1, 0.1]],
             query_range=[[-0.15, 0.15], [-0.15, 0.15]],
             query_type= "simple", #"iterative"
             n_check_invariant=2,
             N_overt=1,
             dt=0.1,
             recalculate=False,
             ncore=N_CORES
             )
example_1.run()
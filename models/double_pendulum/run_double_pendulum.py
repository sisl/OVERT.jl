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
             keras_controller_file="double_pend_nn_controller_ilqr_data.h5",
             overt_dynamics_file="double_pendulum.jl",
            controller_bounding_values=[[-1., 1.], [-1., 1.]],
            integration_map=['s2', 's3', 'o0', 'o1'],
            model_states=[b'th1', b'th2', b'u1', b'u2'],
            model_controls=[b'T1', b'T2'],
            init_range=[[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
            query_range=[[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
            query_type="simple",
            n_check_invariant=2,
            N_overt=1,
            dt=0.02,
            recalculate=False,
             ncore=N_CORES
             )
example_1.run()
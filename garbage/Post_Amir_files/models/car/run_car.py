import os
import sys
assert len(sys.argv) == 3, "you should pass marabou address AND number of cores used in the job"
MARABOU_PATH = sys.argv[1]
N_CORES = int(sys.argv[2])
print(os.getcwd())
sys.path.insert(0, "../../MarabouMC")
sys.path.insert(0, MARABOU_PATH)

from example import OvertMCExample

example_car = OvertMCExample(
            keras_controller_file="/home/amaleki/Downloads/car_controller.h5",
            overt_dynamics_file="car.jl",
            controller_bounding_values=[[-1., 1.], [-1., 1.]],
            integration_map=['o0', 'o1', 'o2', 'c0'],
            model_states=[b'x1', b'x2', b'x3', b'x4'],
            model_controls=[b'c1', b'c2'],
            init_range=[[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
            query_range=[[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
            overt_range=[[-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8]],
            query_type="simple",
            n_check_invariant=5,
            N_overt=2,
            dt=0.02,
            recalculate=True,
            ncore=N_CORES
             )

example_car.run()
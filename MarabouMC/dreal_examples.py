# dreal comparisons
from MC_utilities import NonlinearControlleTRExperiment
from dynamical_systems.dynamical_systems import SinglePendulum
import colored_traceback.always
from transition_systems import NonlinearDynamics

experiment = NonlinearControlleTRExperiment(
            keras_controller_file="../OverApprox/models/single_pend_nn_controller_ilqr_data.h5",
            controller_bounding_values=[[-2., 2.]],
            dynamics_object=NonlinearDynamics(SinglePendulum(), dt=.1),
            query_range=[[-0.3, 0.3], [-0.3, 0.3]],
            init_range=[[-0.1, 0.1], [-0.1, 0.1]],
            n_steps=2,
            )
experiment.run()
import os
import os.path
import sys
assert len(sys.argv) == 3, "you should pass marabou address AND number of cores used in the job"
MARABOU_PATH = sys.argv[1]
N_CORES = int(sys.argv[2])

sys.path.insert(0, "..")
sys.path.insert(0, MARABOU_PATH)

import h5py
from keras.models import load_model

from overt_to_python import OvertConstraint
from transition_systems import KerasController, Dynamics, TFControlledTransitionRelation, TransitionSystem, OvertDynamics, constraint_variable_to_interval
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC

class OvertMCExample():
    def __init__(self,
                 keras_controller_file,
                 overt_dynamics_file,
                 controller_bounding_values,
                 integration_map,
                 model_states,
                 model_controls,
                 init_range,
                 query_range,
                 query_type,
                 n_check_invariant=4,  # number of timestep checked in MC
                 N_overt=2,
                 dt=0.01,
                 recalculate=True
                 ):
        """

        Args:
            keras_controller_file: .h5 file address that is saved keras model.
            overt_dynamics_file: .jl file address that includes dynamics of the problem.
            controller_bounding_values: values at which control inputs are capped. format:
                                    [[u1_min, u1_max], [u2_min, u2_max], ...]
            integration_map: a list that specifies the derivative of each state. use s for state and o for overts
                example: ['s1','s2','o0','o1'] means the ds[0] = s1, ds[1]=s2, ds[2]=overt_output[0], ds[3]= overt_output[1]
            model_states: list of state variables in the model like th1, th2, ...
            model_controls: list of control variables in the model like T1, T2, ...
            init_range: a list of lists that specifies the initial set. format:
                         [[x1_min, x1_max],[x2_min, x2_max], ...] for an initial set x1_min<x1<x1_max and x2_min<x2<x2_max, ...
            query_range: a list of lists that specifies property of interest. format:
                         [[x1_min, x1_max],[x2_min, x2_max], ...] for a property x1_min<x1<x1_max and x2_min<x2<x2_max, ...
            query_type: can take the followings
                        "simple": this is a simple query with variables and constraints growing in each time step.
                        "iterative": this is iterative process where the query is expanded at each time step until
                                     UNSAT is achieved. The number of variables and constraints does not grow.
            n_check_invariant: number of timesteps that should be check in the invariant property
            N_overt: number of intermediate points in the overt alg.
            dt: dt of euler integration.
            recalculate: if True, the overt is reculculated by running Julia. Otherwise, the already saved file is parsed.
        """
        self.keras_controller_file = keras_controller_file
        self.overt_dynamics_file = overt_dynamics_file
        self.controller_bounding_values = controller_bounding_values
        self.integration_map = integration_map
        self.model_states = model_states
        self.model_controls = model_controls
        self.init_range = init_range
        self.query_range = query_range
        self.query_type = query_type
        self.n_check_invariant = n_check_invariant  # number of timestep checked in MC
        self.N_overt = N_overt
        self.dt = dt
        self.recalculate = recalculate

        assert self.overt_dynamics_file.split('.')[-1] == "jl"
        self.dynamic_save_file = self.overt_dynamics_file[:-3] + "_savefile.h5" # OVERT inputs/outputs will be saved here.
        self.overt_dyn_obj = None # will contain an instance of OVERTDynamics class.
        self.controller_obj = None # will contain an instance of KerasConstraint class.
        self.state_vars = None # will contain a list of variables that are assigned to model states
        self.control_vars = None # will contain a list of variables that are assigned to model controls

    def setup_overt_dyn_obj(self):
        if self.recalculate:
            if os.path.exists(self.dynamic_save_file): os.remove(self.dynamic_save_file)
            fid = h5py.File(self.dynamic_save_file, "w")
            fid["overt/N"] = self.N_overt
            fid["overt/states"] = self.model_states
            fid["overt/controls"] = self.model_controls
            fid["overt/bounds/states"] = self.init_range
            fid["overt/bounds/controls"] = self.controller_bounding_values
            fid.close()
            os.system("julia " + self.overt_dynamics_file)
        overt_obj = OvertConstraint(self.dynamic_save_file)
        self.state_vars = overt_obj.state_vars

        dx_vec = []
        for str in self.integration_map:
            indicator = str[0]
            idx = int(str[1])
            if indicator == 's':
                dx_vec.append(self.state_vars[idx])
            elif indicator == 'o':
                dx_vec.append(overt_obj.output_vars[idx])
        self.overt_dyn_obj = OvertDynamics(overt_obj, dx_vec, self.dt)

    def setup_controller_obj(self):
        model = load_model(self.keras_controller_file)
        min_vals = [x[0] for x in self.controller_bounding_values]
        max_vals = [x[1] for x in self.controller_bounding_values]
        self.controller_obj = KerasController(keras_model=model, cap_values=[min_vals, max_vals])

    def setup_property(self):
        prop_list = []
        for i, prop_range in enumerate(self.query_range):
            prop_list += constraint_variable_to_interval(self.state_vars[i], prop_range[0], prop_range[1])
        prop = ConstraintProperty(prop_list)
        return prop

    def _simple_run(self):
        self.setup_controller_obj()

        self.setup_overt_dyn_obj()

        tr = TFControlledTransitionRelation(dynamics_obj=self.overt_dyn_obj, controller_obj=self.controller_obj)
        init_set = dict(zip(self.state_vars, self.init_range))
        ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)
        solver = MarabouWrapper(n_worker=N_CORES)
        prop = self.setup_property()
        algo = BMC(ts=ts, prop=prop, solver=solver)
        return algo.check_invariant_until(self.n_check_invariant)

    def run(self):
        if self.query_type == "simple":
            self._simple_run()
        else:
            raise(NotImplementedError())

if __name__ == "__main__":
    # example_1 = OvertMCExample(
    #              keras_controller_file="../OverApprox/models/single_pend_nn_controller_ilqr_data.h5",
    #              overt_dynamics_file="../OverApprox/models/single_pendulum2.jl",
    #              controller_bounding_values=[[-2., 2.]],
    #              integration_map=['s1', 'o0'],
    #              model_states=[b'th', b'dth'],
    #              model_controls=[b'T'],
    #              init_range=[[-0.1, 0.1], [-0.1, 0.1]],
    #              query_range=[[-0.3, 0.3], [-0.3, 0.3]],
    #              query_type="simple",
    #              n_check_invariant=10,
    #              N_overt=2,
    #              dt=0.1,
    #              recalculate=False
    #              )
   # example_1.run()

    example_2 = OvertMCExample(
                 keras_controller_file="../OverApprox/models/double_pend_nn_controller_ilqr_data.h5",
                 overt_dynamics_file="../OverApprox/models/double_pendulum2.jl",
                 controller_bounding_values=[[-1., 1.], [-1., 1.]],
                 integration_map=['s2', 's3', 'o0', 'o1'],
                 model_states=[b'th1', b'th2', b'u1', b'u2'],
                 model_controls=[b'T1', b'T2'],
                 init_range=[[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
                 query_range=[[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
                 query_type="simple",
                 n_check_invariant=3,
                 N_overt=1,
                 dt=0.02,
                 recalculate=False
                 )
    example_2.run()



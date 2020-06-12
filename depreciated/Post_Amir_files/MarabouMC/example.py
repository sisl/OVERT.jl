import os
import os.path
import warnings
warnings.filterwarnings('ignore')

#import sys
# assert len(sys.argv) == 3, "you should pass marabou address AND number of cores used in the job"
# MARABOU_PATH = sys.argv[1]
# N_CORES = int(sys.argv[2])
#
# sys.path.insert(0, "..")
# sys.path.insert(0, MARABOU_PATH)

import h5py
from keras.models import load_model

from MC_constraints import MaxConstraint
from overt_to_python import OvertConstraint
from transition_systems import KerasController, TFControlledTransitionRelation, TransitionSystem, OvertDynamics, constraint_variable_to_interval
from marabou_interface import MarabouWrapper
from gurobi_interface import GurobiPyWrapper
from properties import ConstraintProperty
from MC_interface import BMC

julia_executable_path = "/Applications/Julia-1.2.app/Contents/Resources/julia/bin/julia"

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
                 overt_range=None,
                 query_type="simple", 
                 n_check_invariant=2,  # number of timestep checked in MC
                 N_overt=2,
                 dt=0.01,
                 recalculate=True,
                 ncore=4
                 ):
        """

        Args:
            keras_controller_file: .h5 file address that is saved keras model for the controller.
            overt_dynamics_file: .jl file address that includes dynamics of the problem.
            controller_bounding_values: values at which control inputs are capped. format:
                                    [[u1_min, u1_max], [u2_min, u2_max], ...]
            integration_map: a list that specifies the derivative of each state. use s for state and o for overts and c for control.
                example: ['s1','s2','o0','c1'] means the ds[0] = s1, ds[1]=s2, ds[2]=overt_output[0], ds[3]= control_var[1]
            model_states: list of state variables in the model like th1, th2, ...
            model_controls: list of control variables in the model like T1, T2, ...
            init_range: a list of lists that specifies the initial set. format:
                         [[x1_min, x1_max],[x2_min, x2_max], ...] for an initial set x1_min<x1<x1_max and x2_min<x2<x2_max, ...
            query_range: a list of lists that specifies property of interest. format:
                         [[x1_min, x1_max],[x2_min, x2_max], ...] for a property x1_min<x1<x1_max and x2_min<x2<x2_max, ...
            overt_range: a list of lists that specifies the set over which overt is valid. format:
                         [[x1_min, x1_max],[x2_min, x2_max], ...] for a set x1_min<x1<x1_max and x2_min<x2<x2_max, ...
                         default value is None. If overt_range is None:
                            - overt_range will be initialized to init_range.
                            - only 1 timestep would be guaranteed valid.

            query_type: can take the followings
                        "simple": this is a simple query with variables and constraints growing in each time step.
                        "iterative": this is iterative process where the query is expanded at each time step until
                                     UNSAT is achieved. The number of variables and constraints does not grow.
            n_check_invariant: number of timesteps that should be check for the invariant property
            N_overt: number of intermediate points in the overt alg.
            dt: dt of euler integration.
            recalculate: if True, the overt is recalculated by running Julia. Otherwise, the already saved file is parsed.
            ncore: #cpus for running marabou in parallel.
        """
        self.keras_controller_file = keras_controller_file
        self.overt_dynamics_file = overt_dynamics_file
        self.controller_bounding_values = controller_bounding_values
        self.integration_map = integration_map
        self.model_states = model_states
        self.model_controls = model_controls
        self.init_range = init_range
        self.query_range = query_range
        self.overt_range = overt_range
        self.query_type = query_type
        self.n_check_invariant = n_check_invariant  # number of timestep checked in MC
        self.N_overt = N_overt
        self.dt = dt
        self.recalculate = recalculate
        self.ncore = ncore


        assert self.overt_dynamics_file.split('.')[-1] == "jl" # making sure there exists a .jl file
        self.dynamic_save_file = self.overt_dynamics_file[:-3] + "_savefile.h5" # OVERT inputs/outputs will be saved here.
        self.overt_dyn_obj = None # will contain an instance of OVERTDynamics class.
        self.controller_obj = None # will contain an instance of KerasConstraint class.
        self.state_vars = None # will contain a list of variables that are assigned to model states
        self.control_vars = None # will contain a list of variables that are assigned to model controls
        self.setup_overt_range()

    def setup_overt_range(self):
        if self.overt_range is None:
            print("using init_range for overt_range")
            assert self.n_check_invariant == 2 or self.query_type != "simple", "simple query  types only support two time steps."
            self.overt_range = self.init_range
        else:
            print("overt_range is prespecified. Make sure the dynamics remain within this range. The program does not automatically check this.")

    def setup_overt_dyn_obj(self):
        if self.recalculate:
            # Set some hyper parameters / initial values
            if os.path.exists(self.dynamic_save_file): os.remove(self.dynamic_save_file)
            fid = h5py.File(self.dynamic_save_file, "w")
            fid["overt/N"] = self.N_overt
            fid["overt/states"] = self.model_states
            fid["overt/controls"] = self.model_controls
            fid["overt/bounds/states"] = self.overt_range
            fid["overt/bounds/controls"] = self.controller_bounding_values
            fid.close()

            # run julia to generate bounds
            print("julia starting ...")
            os.system(julia_executable_path + " " + self.overt_dynamics_file)
            print("julia finished.")
        else:
            print("recalcualte is turned off. using existing overt file.")

        # read bounds from file
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
            elif indicator == 'c':
                dx_vec.append(overt_obj.control_vars[idx])
            else:
                raise(IOError("maping is not recognized"))
        self.overt_dyn_obj = OvertDynamics(overt_obj, dx_vec, self.dt)

    def setup_controller_obj(self):
        model = load_model(self.keras_controller_file)
        min_vals = [x[0] for x in self.controller_bounding_values]
        max_vals = [x[1] for x in self.controller_bounding_values]
        self.controller_obj = KerasController(keras_model=model, cap_values=[min_vals, max_vals])

    def setup_property(self):
        prop_list = []
        prop_output = []
        for i, prop_range in enumerate(self.query_range):
            prop_list += constraint_variable_to_interval(self.state_vars[i], prop_range[0], prop_range[1])
            prop_output.append(self.state_vars[i])
        prop = ConstraintProperty(prop_list, prop_output)
        return prop

    def _simple_run(self, n_check_invariant):
        self.setup_controller_obj()

        self.setup_overt_dyn_obj()
        tr = TFControlledTransitionRelation(dynamics_obj=self.overt_dyn_obj, controller_obj=self.controller_obj, turn_max_to_relu=True)

        init_set = dict(zip(self.state_vars, self.init_range))
        ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)
        for c in ts.transition_relation.constraints:
            assert(not isinstance(c, MaxConstraint))
        #solver = MarabouWrapper(n_worker=self.ncore)
        solver = GurobiPyWrapper()
        prop = self.setup_property()
        algo = BMC(ts=ts, prop=prop, solver=solver)
        return algo.check_invariant_until(n_check_invariant)

    def _iterative_run(self, itr_max=20):
        for n in range(self.n_check_invariant-1):
            print("time step:", n)
            print("init_range:", self.init_range)
            print("query_range:", self.query_range)
            itr = 0
            while itr < itr_max:
                result, value, stats = self._simple_run(2)
                if result.name == "UNSAT":
                    break
                else:
                    value_dict = {k:v for (k,v) in value}
                    self.widen_property(value_dict)
            if itr == itr_max: raise(ValueError("iterations did not converge in %d trials"%itr_max))
            self.update_init_range()

    def update_init_range(self, EPS=1E-3):
        for i in range(len(self.init_range)):
            self.init_range[i] = [self.query_range[i][0]+EPS, self.query_range[i][1]-EPS]

        self.recalculate = True

    def widen_property(self, value, EPS=1E-3, alpha=0.2):
        violation_found = False
        for i in range(len(self.state_vars)):
            x = self.state_vars[i]
            lb, ub = self.query_range[i]
            v = "%s@1"% x
            k = value[v]
            if k - lb < EPS:
                self.query_range[i][0] = lb - (ub - lb) * alpha
                violation_found = True
                break
            elif ub - k < EPS:
                self.query_range[i][1] = ub + (ub - lb) * alpha
                violation_found = True
                break

        assert violation_found, "violation was not found."
        self.recalculate = False


    def run(self):
        if self.query_type == "simple":
            self._simple_run(self.n_check_invariant)
        elif self.query_type == "iterative":
            self._iterative_run()
        else:
            raise(IOError())

if __name__ == "__main__":
    example_1 = OvertMCExample(
                 keras_controller_file="../OverApprox/models/single_pend_nn_controller_ilqr_data.h5",
                 overt_dynamics_file="../OverApprox/models/single_pendulum2.jl",
                 controller_bounding_values=[[-2., 2.]],
                 integration_map=['s1', 'o0'],
                 model_states=[b'th', b'dth'],
                 model_controls=[b'T'],
                 init_range=[[-0.1, 0.1], [-0.1, 0.1]],
                 query_range=[[-0.3, 0.3], [-0.3, 0.3]],
                 query_type="iterative",
                 n_check_invariant=10,
                 N_overt=2,
                 dt=0.1,
                 recalculate=False
                 )
    example_1.run()

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



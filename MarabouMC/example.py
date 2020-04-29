import os
import os.path
import h5py

class OvertMCExample():
    def __init__(self,
                 keras_controller_file,
                 overt_dynamics_file,
                 controller_max_values,
                 integration_map,
                 state_vars,
                 control_vars,
                 init_range,
                 query_range,
                 query_type,
                 n_check_invariant=4,  # number of timestep checked in MC
                 N_overt=2,
                 dt=0.01
                 ):
        """

        Args:
            keras_controller_file: .h5 file address that is saved keras model.
            overt_dynamics_file: .jl file address that includes dynamics of the problem.
            controller_max_values: values at which control inputs are capped. format:
                                    [[u1_min, u1_max], [u2_min, u2_max], ...]
            integration_map: a list that specifies the derivative of each state. use s for state and o for overts
                example: [s1,s2,o0,o1] means the ds[0] = s1, ds[1]=s2, ds[2]=overt_output[0], ds[3]= overt_output[1]
            state_vars: list of state variables
            control_vars: list of control variables
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
        """
        self.keras_controller_file = keras_controller_file
        self.overt_dynamics_file = overt_dynamics_file
        self.controller_max_values = controller_max_values
        self.integration_map = integration_map
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.init_range = init_range
        self.query_range = query_range
        self.query_type = query_type
        self.n_check_invariant = n_check_invariant  # number of timestep checked in MC
        self.N_overt = N_overt
        self.dt = dt

        self.dynamic_save_file = self.overt_dynamics_file.split('.')[0] + "_save.h5" # OVERT inputs/outputs will be saved here.
        self.overt_dyn_obj = None # will contain an instance of OVERTDynamics class.
        self.controller_obj = None # will contain an instance of KerasConstraint class.

    def proprocess_overt(self):
        if os.path.exists(self.dynamic_save_file): os.remove(self.dynamic_save_file)
        fid = h5py.File(self.dynamic_save_file, "w")
        fid["overt/N"] = self.N_overt
        fid["overt/states"] = self.state_vars
        fid["overt/controls"] = self.control_vars
        fid["overt/bounds/states"] = self.init_range
        fid["overt/bounds/controls"] = self.controller_max_values
        fid.close()

    def setup_overt_dyn_obj(self):
        os.system("julia " + self.overt_dynamics_file)
        overt_obj = OvertConstraint(self.overt_dynamics_file)
        dx_vec = []
        for str in self.integration_map:
            if str[0] == 's':
                dx_vec.append(self.state_vars[str[1]])
            elif str[1] == 'o':
                dx_vec.append(overt_obj.output_vars[str[1]])
        self.overt_dyn_obj = OVERTDynamics(overt_obj, dx_vec, self.dt)

    def setup_controller_obj(self):
        model = load_model(self.keras_controller_file)
        self.controller_obj = KerasController(keras_model=model)

    def setup_property(self):
        prop_list = []
        for i, prop_range in enumerate(self.query_range):
            prop_list += setup_interval_constraint(self.state_vars[i], prop_range[0], prop_range[1])
        prop = ConstraintProperty(prop_list)
        return prop

    def simple_call(self):
        tr = TFControlledTransitionRelation(dynamics_obj=self.overt_dyn_obj, controller_obj=self.controller_obj)
        ts = TransitionSystem(states=tr.states, initial_set=self.init_range, transition_relation=tr)
        solver = MarabouWrapper()
        prop = self.setup_property()
        algo = BMC(ts=ts, prop=prop, solver=solver)
        return algo.check_invariant_until(self.n_check_invariant)




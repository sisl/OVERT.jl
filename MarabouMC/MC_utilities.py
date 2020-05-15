# utilities for running experiments with the MC

from transition_systems import KerasController, TFControlledTransitionRelation, TransitionSystem, constraint_variable_to_interval
from properties import ConstraintProperty
from MC_interface import BMC


class ControlledTRExperiment:
    def __init__(self,
                keras_controller_file,
                controller_bounding_values,
                query_range, # invariant box properties
                init_range, # starting set
                algo='BMC',
                n_steps=2,
                dt=0.01)
        # note: dynamics not specified because this can be used
        # with OVERT or with true nonlinear dynamics
        self.keras_controller_file = keras_controller_file
        self.controller_bounding_values = self.controller_bounding_values
        self.query_range = query_range
        self.init_range = init_range
        self.n_steps = n_steps
        self.dt = dt
        self.setup_controller_obj()

    def setup_controller_obj(self):
        model = load_model(self.keras_controller_file)
        min_vals = [x[0] for x in self.controller_bounding_values]
        max_vals = [x[1] for x in self.controller_bounding_values]
        self.controller_obj = KerasController(keras_model=model, cap_values=[min_vals, max_vals])

    def setup_property(self):
        prop_list = []
        for i, prop_range in enumerate(self.query_range):
            prop_list += constraint_variable_to_interval(self.states[i], prop_range[0], prop_range[1])
        prop = ConstraintProperty(prop_list)
        return prop 

class NonlinearControlleTRExperiment(ControlledTRExperiment):
    """
    A class for running model checking experiments using the full nonlinear dynamics.
    Args:
        solver: {'marabou', 'smtlib2', 'dreal'}, 'smtlib2' prints smtlib2 file
    """
    def __init__(self,
                keras_controller_file,
                controller_bounding_values,
                dynamics_object, 
                query_range, # invariant box properties
                init_range,
                algo=BMC,
                n_steps=2,
                dt=0.01,
                solver='smtlib2')
        super().__init__(keras_controller_file,
                        controller_bounding_values,
                        query_range, # invariant box properties
                        init_range,
                        algo=algo,
                        n_steps=n_steps,
                        dt=dt)
        self.dynamics = dynamics_object
        self.setup_solver(solver)

    def run(self):
        tr = TFControlledTransitionRelation(dynamics_obj=self.dynamics, controller_obj=self.controller_obj)
        init_set = dict(zip(tr.states, self.init_range))
        ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)
        solver = self.solver
        prop = self.setup_property()
        algo = self.algo(ts=ts, prop=prop, solver=solver)
        return algo.check_invariant_until(n_check_invariant)

    def setup_solver(self):
        """
        set self.solver. should be an object with the API of MarabouWrapper
        """
        pass
    

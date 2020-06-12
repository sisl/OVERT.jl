from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC
from transition_systems import TransitionSystem, constraint_variable_to_interval,KerasController,TFControlledTransitionRelation

class ReachabilityIterator():
    def __init__(self, controller_keras_obj, overt_obj, init_set_0, alpha=1.05, max_itr=100, ncheck_invariant=2, cap_values=None):
        self.alpha = alpha
        self.ncheck_invariant = ncheck_invariant
        self.max_itr = max_itr
        self.cap_values = cap_values
        self.overt_obj = overt_obj
        self.init_set = init_set_0
        self.init_history = [init_set_0]
        self.states = overt_obj.states.reshape(-1).tolist()
        self.controller_keras_obj = controller_keras_obj
        self.clean()

    def clean(self):
        self.prop_list = []
        self.prop_set = {}
        self.solver = MarabouWrapper()


    def update_transition_system(self):
        controller = KerasController(keras_model=self.controller_keras_obj, cap_values=self.cap_values)
        tr = TFControlledTransitionRelation(dynamics_obj=self.overt_obj,
                                            controller_obj=controller)
        self.ts = TransitionSystem(states=tr.states, initial_set=self.init_set, transition_relation=tr)

    def get_prop_lb(self, lb):
        if lb < -0.1:
            lb_prop = lb*self.alpha
        elif lb > 0.1 :
            lb_prop = lb/self.alpha
        else:
            lb_prop = lb - 0.2
        return lb_prop

    def get_prop_ub(self, ub):
        if ub < -0.1:
            ub_prop = ub/self.alpha
        elif ub > 0.1 :
            ub_prop = ub*self.alpha
        else:
            ub_prop = ub + 0.2
        return ub_prop

    def setup_property(self, no_initial_expansion=False):
        for x in self.init_set.keys():
            lb, ub = self.init_set[x]
            if no_initial_expansion:
                lb_prop = lb - 0.01
                ub_prop = ub + 0.01
            else:
                lb_prop = self.get_prop_lb(lb)
                ub_prop = self.get_prop_ub(ub)

            self.prop_set[x] = (lb_prop, ub_prop)
            self.prop_list += constraint_variable_to_interval(x, lb_prop, ub_prop)

    def widen_property(self, value, EPS=1E-5):
        for i, x in enumerate(self.states):
            v = "%s@%d" % (x, self.ncheck_invariant-1)
            k = value[v]
            if k - self.prop_set[x][0] < EPS:
                lb_prop = self.get_prop_lb(self.prop_set[x][0])
                ub_prop = self.prop_set[x][1]
                self.prop_set[x] = (lb_prop, ub_prop)
                self.prop_list[2 * i] = constraint_variable_to_interval(x, lb_prop, ub_prop)[0]
                #print("expanding property of %s to "%x, self.prop_set[x])
                return

            if self.prop_set[x][1] - k < EPS:
                lb_prop = self.prop_set[x][0]
                ub_prop = self.get_prop_ub(self.prop_set[x][1])
                self.prop_set[x] = (lb_prop, ub_prop)
                self.prop_list[2 * i + 1] = constraint_variable_to_interval(x, lb_prop, ub_prop)[1]
                #print("expanding property of %s to " % x, self.prop_set[x])
                return

        # it should not get to here
        raise(ValueError("Widen Property did not work."))

    def update_init_set(self):
        self.init_set = self.prop_set.copy()
        self.init_history.append(self.init_set)
        self.clean()

    def run_one_timestep(self):
        itr = 0
        self.setup_property(no_initial_expansion=True)
        while itr < self.max_itr:
            #self.history.append("trying, init_set: " + str(self.init_set) + "prop_set: "+str(self.prop_set))
            self.update_transition_system()
            itr += 1
            prop = ConstraintProperty(self.prop_list)
            algo = BMC(ts=self.ts, prop=prop, solver=self.solver)

            for x, x_int in self.prop_set.items():
                print("    ", x, ": ", x_int)

            result, value, stats = algo.check_invariant_until(self.ncheck_invariant)


            result = result.name
            if result == "UNSAT":
                return
            else:
                self.widen_property(dict(list(value)))

        if itr == self.max_itr:
            raise(ValueError("max iteration reached."))

    def run(self, ntime):
        for i in range(ntime):
            #self.history.append("running time step %d" %i)
            if i > 0: self.update_init_set()
            self.run_one_timestep()
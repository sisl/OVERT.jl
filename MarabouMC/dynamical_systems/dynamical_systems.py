# sample dynamical systems
import numpy as np
from MC_constraints import NLConstraint, Constraint, Monomial

def f1(state, control):
    # 2D state, 1D control
    x = state[0]
    theta = state[1]
    dt = 0.01
    # state_tp1 = f(state_t, control_t)
    return state + (x**3)*(np.sin(theta)**2 + np.exp(control))*dt

class SinglePendulum:   #"T + sin(theta) - 0.2*theta_dot"
    def __init__(self):
        self.control_inputs = ['T']
        self.states = np.array(['theta', 'theta_dot'], dtype=object)
        self.dx = ['theta_dot', 'theta_double_dot']
        # the constraints that define the dx variables. 
        # "theta_double_dot = T + sin(theta) - 0.2*theta_dot"
        self.dx_constraints = [NLConstraint('EQUALITY', out='v1', fun="sin", indep_var = "theta"),
                               Constraint('EQUALITY', monomials=[Monomial(1.0, "T"), 
                                                                Monomial(1.0, "v1"),
                                                                Monomial(-0.2, "theta_dot"),
                                                                Monomial(-1.0, "theta_double_dot")
                                                                ], scalar=0)]
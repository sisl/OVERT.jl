# sample dynamical systems
import numpy as np
from MC_constraints import NLConstraint

def f1(state, control):
    # 2D state, 1D control
    x = state[0]
    theta = state[1]
    dt = 0.01
    # state_tp1 = f(state_t, control_t)
    return state + (x**3)*(np.sin(theta)**2 + np.exp(control))*dt

class SinglePendulum:
    def __init__(self):
        self.contol_inputs = ['T']
        self.states = ['theta', 'theta_dot']
        self.dx = ['theta_dot', 'theta_double_dot']
        # the constraints that define the dx variables. 
        self.dx_constraints = [NLConstraint('EQUALITY', left='theta_double_dot', 
                                     right="T + sin(theta) - 0.2*theta_dot",
                                     indep_vars = ["T", "theta", "theta_dot"])
                              ]
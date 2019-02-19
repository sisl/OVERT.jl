import numpy as np

# Define the safe sets here, bounds will be inverted later (to hopefully get to
# the result that reaching the complements of these sets is unstaisfiable)

# TODO: I think I may only need 1 bound on tdlb and tdub outputs. e.g. 

# multiple steps, start small, don't get too big
def bounds_1():
    inputs_min = {"theta_0": 0*np.pi/180,
                "theta_dot_0": -0.01*np.pi/180,
                "theta_dot_hats": -1000*np.pi/180
    }
    inputs_max = {"theta_0": 0*np.pi/180,
                "theta_dot_0": 0.01*np.pi/180,
                "theta_dot_hats": 1000*np.pi/180
    }
    outputs_min = {"thetas": -23*np.pi/180,
                    "theta_dot_hats": -1000*np.pi/180,
                    "tdlbs": -10000*np.pi/180,
                    "tdubs": -10000*np.pi/180
    }
    outputs_max = {"thetas": 23*np.pi/180,
                    "theta_dot_hats": 1000*np.pi/180,
                    "tdlbs": 10000*np.pi/180,
                    "tdubs": 10000*np.pi/180
    }
    return (inputs_min, inputs_max, outputs_min, outputs_max)

# sanity check
# start very close to falling over. 
# Expect that we cannot recover
def difficult():
    inputs_min = {"theta_0": 21*np.pi/180,
                "theta_dot_0": -0.01*np.pi/180,
                "theta_dot_hats": -1000*np.pi/180
    }
    inputs_max = {"theta_0": 22*np.pi/180,
                "theta_dot_0": 0.01*np.pi/180,
                "theta_dot_hats": 1000*np.pi/180
    }
    outputs_min = {"thetas": -23*np.pi/180,
                    "theta_dot_hats": -1000*np.pi/180,
                    "tdlbs": -10000*np.pi/180,
                    "tdubs": -10000*np.pi/180
    }
    outputs_max = {"thetas": 23*np.pi/180,
                    "theta_dot_hats": 1000*np.pi/180,
                    "tdlbs": 10000*np.pi/180,
                    "tdubs": 10000*np.pi/180
    }
    return (inputs_min, inputs_max, outputs_min, outputs_max)

# sanity check. Start very close to upside down and then expect the pendulum to be upright in 1 timestep
def impossible():
    inputs_min = {"theta_0": 178*np.pi/180,
                "theta_dot_0": -0.01*np.pi/180,
                "theta_dot_hats": -1000*np.pi/180
    }
    inputs_max = {"theta_0": 179*np.pi/180,
                "theta_dot_0": 0.01*np.pi/180,
                "theta_dot_hats": 1000*np.pi/180
    }
    outputs_min = {"thetas": -1*np.pi/180,
                    "theta_dot_hats": -1000*np.pi/180,
                    "tdlbs": -10000*np.pi/180,
                    "tdubs": -10000*np.pi/180
    }
    outputs_max = {"thetas": 1*np.pi/180,
                    "theta_dot_hats": 1000*np.pi/180,
                    "tdlbs": 10000*np.pi/180,
                    "tdubs": 10000*np.pi/180
    }
    return (inputs_min, inputs_max, outputs_min, outputs_max)

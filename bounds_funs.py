import numpy as np

# Define the safe sets here, bounds will be inverted later (to hopefully get to
# the result that reaching the complements of these sets is unstaisfiable)

# TODO: I think I may only need 1 bound on tdlb and tdub outputs. e.g. 
class bounds():
    def __init__(self):
        self.inputs_min = {}
        self.inputs_max = {}
        self.outputs_min = {}
        self.outputs_max = {}

class pendulum_bounds(bounds):
    def __init__(self):
        super().__init__()
        self.populate_unimportant_bounds()

    def populate_unimportant_bounds(self):
        # inputs: 
        self.inputs_min["theta_dot_hats"] = -1000.
        self.inputs_max["theta_dot_hats"] = 1000.
        # outputs:
        # max
        # self.outputs_max["tdlbs"] = 1000.
        # self.outputs_max["tdubs"] = 1000.
        # #  mins
        # self.outputs_min["tdlbs"] = -1000.
        # self.outputs_min["tdubs"] = -1000.

# multiple steps, start small, don't get too big
# unsat for both random and real controller
def bounds_1():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = 0*np.pi/180
    bounds.inputs_max["theta_0"] = 0*np.pi/180
    bounds.inputs_min["theta_dot_0"] = -0.01*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 0.01*np.pi/180
    #
    bounds.outputs_min["thetas"] = -23*np.pi/180
    bounds.outputs_max["thetas"] = 23*np.pi/180
    #
    return bounds

# stay in the same set
# for 2-step case
# only check second step (because demorgan!!)
def bounds_2():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = -10*np.pi/180
    bounds.inputs_max["theta_0"] = 10*np.pi/180
    bounds.inputs_min["theta_dot_0"] = 0. 
    bounds.inputs_max["theta_dot_0"] = 0. 
    bounds.outputs_min["theta_2"] = -10*np.pi/180 #- 1e-3*np.pi/180
    bounds.outputs_max["theta_2"] = 10*np.pi/180 #+ 1e-3*np.pi/180
    # theta_1 within bounds if these asserts pass:
    if bounds.inputs_max["theta_0"] + 0.05*bounds.inputs_max["theta_dot_0"] <= bounds.outputs_max["theta_2"]:
        print("theta1 safe")
    else:
        print("theta1 not safe by design: max")
    if bounds.inputs_min["theta_0"] + 0.05*bounds.inputs_min["theta_dot_0"] >= bounds.outputs_min["theta_2"]:
        print("theta1 safe")
    else:
        print("theta1 not safe by design: min")

    return bounds

def bounds_2_5():
    bounds = pendulum_bounds()
    # inputs
    t02 = 10
    bounds.inputs_min["theta_0"] = -t02*np.pi/180
    bounds.inputs_max["theta_0"] = t02*np.pi/180
    bounds.inputs_min["theta_dot_0"] = -60*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 60*np.pi/180
    # outputs
    # thetas
    bounds.outputs_min["theta_1"] = \
            bounds.inputs_min["theta_0"] \
            + 0.1*bounds.inputs_min["theta_dot_0"]
    bounds.outputs_max["theta_1"] = \
            bounds.inputs_max["theta_0"] \
            + 0.1*bounds.inputs_max["theta_dot_0"]
    bounds.outputs_min["theta_2"] = -t02*np.pi/180
    bounds.outputs_max["theta_2"] = t02*np.pi/180
    # theta dots
    bounds.outputs_min["theta_dot_1"] = -120*np.pi/180
    bounds.outputs_max["theta_dot_1"] = 120*np.pi/180
    bounds.outputs_min["theta_dot_2"] = -60*np.pi/180
    bounds.outputs_max["theta_dot_2"] = 60*np.pi/180
    #
    return bounds

def bounds_fixed_start():
    bounds = pendulum_bounds()
    tsafe = 10
    tdsafe = 120
    # inputs
    bounds.inputs_min["theta_0"] = (1)*np.pi/180
    bounds.inputs_max["theta_0"] = (1)*np.pi/180
    bounds.inputs_min["theta_dot_0"] = 0*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 0*np.pi/180
    # outputs
    # thetas
    bounds.outputs_min["theta_1"] = -tsafe*np.pi/180
    bounds.outputs_max["theta_1"] = tsafe*np.pi/180
    bounds.outputs_min["theta_2"] = -tsafe*np.pi/180
    bounds.outputs_max["theta_2"] = tsafe*np.pi/180
    # theta dots
    bounds.outputs_min["theta_dot_1"] = -tdsafe*np.pi/180
    bounds.outputs_max["theta_dot_1"] = tdsafe*np.pi/180
    bounds.outputs_min["theta_dot_2"] = -tdsafe*np.pi/180
    bounds.outputs_max["theta_dot_2"] = tdsafe*np.pi/180
    #
    return bounds

def bounds_3_5():
    bounds = pendulum_bounds()
    # theta dots
    bounds.inputs_min["theta_dot_0"] = -60*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 60*np.pi/180
    bounds.outputs_min["theta_dot_1"] = -120*np.pi/180
    bounds.outputs_max["theta_dot_1"] = 120*np.pi/180
    bounds.outputs_min["theta_dot_2"] = -60*np.pi/180
    bounds.outputs_max["theta_dot_2"] = 60*np.pi/180
    bounds.outputs_min["theta_dot_3"] = -60*np.pi/180
    bounds.outputs_max["theta_dot_3"] = 60*np.pi/180
    #
    # thetas
    bounds.inputs_min["theta_0"] = -5*np.pi/180
    bounds.inputs_max["theta_0"] = 5*np.pi/180
    bounds.outputs_min["theta_1"] = \
            bounds.inputs_min["theta_0"] \
            + 0.1*bounds.inputs_min["theta_dot_0"]
    bounds.outputs_max["theta_1"] = \
            bounds.inputs_max["theta_0"] \
            + 0.1*bounds.inputs_max["theta_dot_0"]
    bounds.outputs_min["theta_2"] = -12*np.pi/180
    bounds.outputs_max["theta_2"] = 12*np.pi/180
    bounds.outputs_min["theta_3"] = -5*np.pi/180
    bounds.outputs_max["theta_3"] = 5*np.pi/180
    #
    return bounds

def bounds_5_5():
    bounds = pendulum_bounds()
    # theta dots
    tds = [60, 120, 60, 60, 60, 60]
    bounds.inputs_min["theta_dot_0"] = -tds[0]*np.pi/180
    bounds.inputs_max["theta_dot_0"] = tds[0]*np.pi/180
    for i in range(1,6):
        bounds.outputs_min["theta_dot_"+str(i)] = -tds[i]*np.pi/180
        bounds.outputs_max["theta_dot_"+str(i)] = tds[i]*np.pi/180
    #
    # thetas
    ts = [3, "NA", 14, 12, 12, 3]
    bounds.inputs_min["theta_0"] = -ts[0]*np.pi/180
    bounds.inputs_max["theta_0"] = ts[0]*np.pi/180
    bounds.outputs_min["theta_1"] = \
            bounds.inputs_min["theta_0"] \
            + 0.1*bounds.inputs_min["theta_dot_0"]
    bounds.outputs_max["theta_1"] = \
            bounds.inputs_max["theta_0"] \
            + 0.1*bounds.inputs_max["theta_dot_0"]
    for i in range(2,6):
        bounds.outputs_min["theta_"+str(i)] = -ts[i]*np.pi/180
        bounds.outputs_max["theta_"+str(i)] = ts[i]*np.pi/180
    #
    return bounds


def bounds_4_5():
    bounds = pendulum_bounds()
    # theta dots
    tds = [120, 240, 120, 120, 120]
    bounds.inputs_min["theta_dot_0"] = -tds[0]*np.pi/180
    bounds.inputs_max["theta_dot_0"] = tds[0]*np.pi/180
    for i in range(1,5):
        bounds.outputs_min["theta_dot_"+str(i)] = -tds[i]*np.pi/180
        bounds.outputs_max["theta_dot_"+str(i)] = tds[i]*np.pi/180
    #
    # thetas
    ts = [20, "NA", 20, 20, 20]
    bounds.inputs_min["theta_0"] = -ts[0]*np.pi/180
    bounds.inputs_max["theta_0"] = ts[0]*np.pi/180
    bounds.outputs_min["theta_1"] = \
            bounds.inputs_min["theta_0"] \
            + 0.1*bounds.inputs_min["theta_dot_0"]
    bounds.outputs_max["theta_1"] = \
            bounds.inputs_max["theta_0"] \
            + 0.1*bounds.inputs_max["theta_dot_0"]
    for i in range(2,5):
        bounds.outputs_min["theta_"+str(i)] = -ts[i]*np.pi/180
        bounds.outputs_max["theta_"+str(i)] = ts[i]*np.pi/180
    #
    return bounds

def bounds_3():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = -15*np.pi/180
    bounds.inputs_max["theta_0"] = 15*np.pi/180
    bounds.inputs_min["theta_dot_0"] = 0. 
    bounds.inputs_max["theta_dot_0"] = 0. 
    bounds.outputs_min["theta_2"] = -15*np.pi/180 
    bounds.outputs_max["theta_2"] = 15*np.pi/180 
    # theta_1 within bounds if these asserts pass:
    if bounds.inputs_max["theta_0"] + 0.05*bounds.inputs_max["theta_dot_0"] <= bounds.outputs_max["theta_2"]:
        print("theta1 safe")
    else:
        print("theta1 not safe by design: max")
    if bounds.inputs_min["theta_0"] + 0.05*bounds.inputs_min["theta_dot_0"] >= bounds.outputs_min["theta_2"]:
        print("theta1 safe")
    else:
        print("theta1 not safe by design: min")

    return bounds

# sanity check
# start very close to falling over. 
# only a good controller can recover
def difficult():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = 21*np.pi/180
    bounds.inputs_max["theta_0"] = 22*np.pi/180
    bounds.inputs_min["theta_dot_0"] = -100*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 100*np.pi/180
    #
    bounds.outputs_min["thetas"] = -23*np.pi/180
    bounds.outputs_max["thetas"] = 23*np.pi/180
    #
    return bounds


# if input bounds are fixed to the same value, it can get stuck in preprocessing and declare UNSAT when it should really declare SAT
def practically_impossible():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = 89*np.pi/180 
    bounds.inputs_max["theta_0"] = 90*np.pi/180 
    bounds.inputs_min["theta_dot_0"] = -0.01*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 0.01*np.pi/180
    #
    bounds.outputs_min["theta_2"] = -1.0*np.pi/180
    bounds.outputs_max["theta_2"] = 1.0*np.pi/180
    #
    return bounds
    
# sanity check. Start very close to upside down and then expect the pendulum to be upright in 1 timestep
# update: produces SAT for both real and random controller, as expected
def impossible():
    bounds = pendulum_bounds()
    bounds.inputs_min["theta_0"] = 178*np.pi/180
    bounds.inputs_max["theta_0"] = 179*np.pi/180
    bounds.inputs_min["theta_dot_0"] = 0.*np.pi/180
    bounds.inputs_max["theta_dot_0"] = 0.*np.pi/180
    #
    bounds.outputs_min["thetas"] = -1*np.pi/180
    bounds.outputs_max["thetas"] = 1*np.pi/180
    #
    return bounds

def minimal_1():
    bounds = pendulum_bounds()
    # outputs
    bounds.outputs_min["thetas"] = -180*np.pi/180
    bounds.outputs_max["thetas"] = 180*np.pi/180
    return bounds

def minimal_2():
    bounds = pendulum_bounds()
    # outputs
    return bounds


    

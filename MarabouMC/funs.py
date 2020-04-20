import numpy as np

def single_pendulum(th, dth, T, dt):
    thnew = th + dth*dt
    ddth = T + np.sin(th) - 0.2*dth
    dthnew = dth + ddth*dt
    return thnew, dthnew
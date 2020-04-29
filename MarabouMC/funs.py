import numpy as np

def single_pendulum(x, T, dt):
    th, dth = x
    thnew = th + dth*dt
    ddth = T + np.sin(th) - 0.2*dth
    dthnew = dth + ddth*dt
    return thnew, dthnew

def Ex_2(x, c, dt):
    x1, x2 = x
    x1new = x1 + x2*dt
    x2new = x2 + (c*x2**2 - x1)*dt
    return x1new, x2new

def car(x, u, dt):
    lr = 1.5
    lf = 1.8
    x1, x2, x3, x4 = x
    u1, u2 = u
    arg = lr/(lr+lf)*np.tan(u2)
    beta = np.arctan(arg)
    dx1 = x4 * np.cos(x3 + beta)
    dx2 = x4 * np.sin(x3 + beta)
    dx3 = 1/lr * x4 * np.sin(beta)
    dx4 = u1
    x1new = x1 + dx1 * dt
    x2new = x2 + dx2 * dt
    x3new = x3 + dx3 * dt
    x4new = x4 + dx4 * dt
    return x1new, x2new, x3new, x4new

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_car(x_vec, lf=1.8, lr=1.5):
    l = lr + lf
    wheels_width, wheels_height = l*0.2, l*0.1
    fig, ax = plt.subplots(figsize=(10, 8))
    for x, y, psi, v in x_vec:
        a, delta_f = u

        x_head, y_head = x + lf * np.cos(psi), y + lf * np.sin(psi)
        x_tail, y_tail = x - lr * np.cos(psi), y - lr * np.sin(psi)

        back_while = patches.Rectangle((x_tail, y_tail), wheels_width, wheels_height, angle=psi*180/np.pi)
        front_while = patches.Rectangle((x_head, y_head), wheels_width, wheels_height, angle=(psi+delta_f)*180/np.pi)


        ax.cla()
        ax.add_patch(back_while)
        ax.add_patch(front_while)
        ax.plot([x_tail, x_head], [y_tail, y_head])
        ax.set_xlim([-3*l, 3*l])
        ax.set_ylim([-3*l, 3*l])

        plt.draw()
        plt.pause(0.01)
#
# x = [9.5,-4.5,2.1,1.5]
# from keras.models import  load_model
# controller_file_name = "/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_10/neural_network_controller_1_keras.h5"
# model = load_model(controller_file_name)
# x_vec = []
# for i in range(50):
#     x_vec.append(x)
#     u = model.predict(np.array(x).reshape(1,-1)).reshape(-1)
#     u -= 20.
#     x = car(x, u, 0.2)
#
#
# plot_car(x_vec)





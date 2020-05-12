import os
import pickle
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display
from matplotlib.animation import FuncAnimation

class CarEnv():
    """
        this is the bicycle model for a car
    """
    def __init__(self,
                 lf=1.8, lr=1.5,  # front and rear lengths
                 x_0=0.,  # initializations
                 dt=0.01, # computational parameters.
                 ):
        self.n_state, self.n_action = 4, 2
        self.lf, self.lr = lf, lr
        self.dt = dt
        if hasattr(x_0, "__len__"):
            self.x_0 = x_0
        else:
            self.x_0 = np.ones(self.n_state) * x_0

        self.x = None
        self.history = None
        self.reset()

    def reset(self):
        self.x = np.array(self.x_0)
        self.history = [{"x": self.x.copy(), "a": [0.] * (self.n_action), "r": 0.}]


    def dynamics(self, states, t, actions):
        x1, x2, x3, x4 = states
        u1, u2 = actions
        arg = self.lr / (self.lr + self.lf) * np.tan(u2)
        beta = np.arctan(arg)
        dx1 = x4 * np.cos(x3 + beta)
        dx2 = x4 * np.sin(x3 + beta)
        dx3 = 1 / self.lr * x4 * np.sin(beta)
        dx4 = u1
        dXdt = np.array([dx1, dx2, dx3, dx4])
        return dXdt

    def step(self, actions):
        #N = 2
        #t = np.linspace(0, self.dt, N)
        #self.x = odeint(self.dynamics, self.x, t, args=(actions, ))[-1, :]
        self.x += self.dt * self.dynamics(self.x, None, actions)
        self.history.append({"x": self.x.copy(), "r": 0, "a": actions})
        return self.x, 0, False, {}

    def save_history(self, name="history.pkl"):
        with open(name, 'wb') as fid:
            pickle.dump(self.history, fid, protocol=pickle.HIGHEST_PROTOCOL)

    def write_message(self, msg):
        self.history[-1]["m"] = msg

    def render(self, mode="human", skip=1, x_goals=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        l = self.lr + self.lf
        wheels_width, wheels_height, screen_fact = l * 0.2, l * 0.1, 5
        for i in range(0, len(self.history), skip):
            x, y, psi, v = self.history[i]["x"]
            a, delta_f = self.history[i]["a"]

            x_head, y_head = x + self.lf * np.cos(psi), y + self.lf * np.sin(psi)
            x_tail, y_tail = x - self.lr * np.cos(psi), y - self.lr * np.sin(psi)

            back_while = patches.Rectangle((x_tail, y_tail), wheels_width, wheels_height, angle=psi * 180 / np.pi)
            front_while = patches.Rectangle((x_head, y_head), wheels_width, wheels_height,
                                            angle=(psi + delta_f) * 180 / np.pi)

            ax.cla()
            ax.add_patch(back_while)
            ax.add_patch(front_while)
            ax.plot([x_tail, x_head], [y_tail, y_head])
            ax.set_xlim([-1, screen_fact * l])
            ax.set_ylim([-screen_fact * l, 2])

            plt.draw()
            plt.pause(0.01)
            if "m" in self.history[i].keys():
                xtext, ytest = 0.5, 0.1
                ax.text(xtext, ytest, self.history[i]["m"], horizontalalignment='center', c='red')

    def animate(self, file_name="airplane.gif", skip=1, x_goal=None, dpi=30, x_goals=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        l = self.lr + self.lf
        wheels_width, wheels_height = l * 0.2, l * 0.1
        x, y, psi, v = self.history[0]["x"]
        a, delta_f = self.history[0]["a"]

        x_head, y_head = x + self.lf * np.cos(psi), y + self.lf * np.sin(psi)
        x_tail, y_tail = x - self.lr * np.cos(psi), y - self.lr * np.sin(psi)

        back_wheel = patches.Rectangle((x_tail, y_tail), wheels_width, wheels_height, angle=psi * 180 / np.pi)
        front_wheel = patches.Rectangle((x_head, y_head), wheels_width, wheels_height,
                                        angle=(psi + delta_f) * 180 / np.pi)
        ax.add_patch(back_wheel)
        ax.add_patch(front_wheel)
        body = ax.plot([x_tail, x_head], [y_tail, y_head])
        ax.set_xlim([-3. * l, 3. * l])
        ax.set_ylim([-3. * l, 3. * l])

        def update(i):
            label = 'timestep {0}'.format(i)
            print(label)
            x, y, psi, v = self.history[i]["x"]
            a, delta_f = self.history[i]["a"]
            wheels_width, wheels_height = l * 0.2, l * 0.1

            x_head, y_head = x + self.lf * np.cos(psi), y + self.lf * np.sin(psi)
            x_tail, y_tail = x - self.lr * np.cos(psi), y - self.lr * np.sin(psi)

            back_wheel.set_xy((x_tail, y_tail))
            back_wheel.set_angle(psi * 180 / np.pi)
            front_wheel.set_xy((x_head, y_head))
            front_wheel.set_angle((psi + delta_f) * 180 / np.pi)
            body.set_x([x_tail, x_head])
            body.set_y([y_tail, y_head])

            return back_wheel, front_wheel, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.history), skip), interval=20)
        if file_name is not None:
            anim.save(file_name, dpi=dpi, writer='imagemagick')
        plt.show()


if __name__ == '__main__':
    env = CarEnv(dt=0.2, x_0=[9.6,  -4.5, 2.117,  1.5])
    env.reset()

    from keras.models import load_model
    controller_file_name = "/home/amaleki/Downloads/car_linear.h5"#/home/amaleki/Downloads/Neural-Network-Controller-Verification-Benchmarks-HSCC-2019-master/Benchmarks/Ex_10/neural_network_controller_1_keras.h5"
    model = load_model(controller_file_name)

    for i in range(5):
        u = model.predict(np.array(env.x).reshape(1, -1)).reshape(-1)
        env.step(u)

    print(env.history)
    env.render()


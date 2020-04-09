import gym
import pickle
import numpy as np
from numpy import sin, cos, pi
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc

USE_LATEX = False
if USE_LATEX:
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)


class Pendulum(gym.Env):
    """
        this is a n-link pendulum with massless links and identical joint mass and link length.
        There is viscous friction assumed. Are joints are actuated. The inputs are
            -m: mass of each point mass
            -L: length of each link
            -c: coefficient of viscous friction
            -x_0: initial angles and initial angular velocities.
            -T: torque at joints

            For the governing equation, it is assumed
            - the angle of each link is measured from the vertical axis.
            - when the pendulum is upright. x[0:1] is [0]*n_state
            - positive y axis is looking downward. hence g= POSITIVE 9.8
    """
    def __init__(self, n_pend,
                 m=0.5, L=0.5, c=0.0, g=9.8,  # physical paramters
                 max_action=1000, x_0=0.,  # initializations
                 integration_method="2st", dt=0.001 # computational parameters.
                 ):
        self.m, self.L, self.c = m, L, c
        self.g, self.dt, self.max_action = g, dt, max_action

        self.n_state = n_pend*2
        if hasattr(x_0, "__len__"):
            self.x_0 = x_0
        else:
            self.x_0 = np.ones(self.n_state) * x_0

        self.init_guess = np.zeros(self.n_state//2)
        self.integration_method = integration_method
        self.x, self.current_guess, self.history = [None] * 3  # these will be updated in the reset.

        self.reset()

    def reset(self):
        self.x = np.array(self.x_0)
        self.current_guess = self.init_guess
        self.history = [{"x": self.x.copy(), "a": [0.] * (self.n_state // 2), "r": 0.}]

    def setup_equation(self, u_prime, x, torques):
        pass

    def step(self, actions):
        actions = np.clip(actions, -self.max_action, self.max_action).tolist()

        if self.integration_method == "1st":
            u_prime = fsolve(self.setup_equation, self.current_guess, args=(self.x, actions))
        else:
            u_prime_half = fsolve(self.setup_equation, self.current_guess, args=(self.x, actions))
            u_half = self.x[self.n_state//2:] + 1 / 2 * self.dt * u_prime_half
            th_half = self.x[:self.n_state//2] + 1 / 2 * self.dt * u_half - 1 / 8 * u_prime_half * self.dt ** 2
            x_half = np.concatenate((th_half, u_half))
            u_prime = fsolve(self.setup_equation, self.current_guess, args=(x_half, actions))

        self.x[self.n_state//2:] += self.dt * u_prime
        self.x[:self.n_state//2] += self.dt * self.x[self.n_state//2:] - 1 / 2 * u_prime * self.dt ** 2
        costs = np.linalg.norm(self.x) + np.linalg.norm(actions)
        self.history.append({"x": self.x.copy(), "r": -costs, "a": actions})
        return self.x, -costs, False, {}

    def save_history(self, name="history.pkl"):
        with open(name, 'wb') as fid:
            pickle.dump(self.history, fid, protocol=pickle.HIGHEST_PROTOCOL)

    def write_message(self, msg):
        self.history[-1]["m"] = msg

    def render(self, mode="human", read_file=False, close=False, skip=1):
        if read_file:
            with open("history.pkl", "rb") as fid:
                self.history = pickle.load(fid)

        plt.figure()
        for i in range(0, len(self.history), skip):

            # computing coordinates
            x = self.history[i]["x"]
            th = x[:self.n_state//2]
            u = x[self.n_state//2:]
            reward = self.history[i]["r"]
            torque = self.history[i]["a"]

            # remember th is measured from topright position. (hence the negative signs below).
            # plotting links
            plt.cla()

            bottom_x, bottom_y = 0, 0
            for j in range(self.n_state // 2):
                top_x, top_y = bottom_x - self.L * sin(th[j]), bottom_y - self.L * cos(th[j])
                if j == 0:
                    plt.plot([bottom_x, top_x], [bottom_y, top_y], color='black')
                    if self.n_state == 2: # single pendulum
                        plt.plot([top_x], [top_y], markersize=10, markerfacecolor='red', marker="o")
                else:
                    plt.plot([bottom_x, top_x], [bottom_y, top_y], color='black',
                        markersize=10, markerfacecolor='red', marker="o")
                bottom_x, bottom_y = top_x, top_y

            # plotting g arrow
            limits = self.L * (self.n_state // 2) * 1.1
            xarrow, yarrow = -0.9 * limits, 0
            plt.arrow(xarrow, yarrow, limits * 0, limits * 0.2, width=0.02, head_width=0.06, color="black")
            plt.xlim([-limits, limits])
            plt.ylim([limits, -limits])
            plt.text(xarrow * 0.95, yarrow + limits * 0.2, "g")

            # typing state inputs.
            xtext, ytest = 0 * limits, -1.25 * limits
            plt.text(xtext, ytest, "t=%d"%i, horizontalalignment='center')
            xtext, ytest = 0 * limits, -1.15 * limits
            str_format = "$th=("  + "%0.1f," * (self.n_state // 2) + \
                         "), u=(" + "%0.1f," * (self.n_state // 2) + \
                         "), T=(" + "%0.1f," * (self.n_state // 2) + "), r=%0.1f$"
            plt.text(xtext, ytest, str_format % tuple(x.tolist() + torque + [reward]), horizontalalignment='center')
            if "m" in self.history[i].keys():
                xtext, ytest = 0 * limits, -1.05 * limits
                plt.text(xtext, ytest, self.history[i]["m"], horizontalalignment='center', c='red')

            plt.draw()
            plt.pause(0.01)
        if close:
            plt.close()

    def animate(self, read_file=False, skip=1, file_name="pend.gif", dpi=30):
        if read_file:
            with open("history.pkl", "rb") as fid:
                self.history = pickle.load(fid)

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        # Plot a scatter that persists (isn't redrawn) and the initial line.
        x = self.history[0]["x"]
        th = x[:self.n_state//2]
        pendulum1_x, pendulum1_y = - self.L * sin(th[0]), - self.L * cos(th[0])
        pendulum2_x, pendulum2_y = pendulum1_x - self.L * sin(th[1]), pendulum1_y - self.L * cos(th[1])

        line1, = ax.plot([0, pendulum1_x], [0, pendulum1_y], color='black')
        line2, = ax.plot([pendulum1_x, pendulum2_x], [pendulum1_y, pendulum2_y],
                         color='black', markersize=10,
                         markerfacecolor='red', marker="o")
        limits = self.L * 2.2
        plt.xlim([-limits, limits])
        plt.ylim([limits, -limits])

        def update(i):
            label = 'timestep {0}'.format(i)
            print(label)
            x = self.history[i]["x"]
            th = x[:2]
            u = x[2:]
            reward = self.history[i]["r"]
            torque = self.history[i]["a"]

            pendulum1_x, pendulum1_y = - self.L * sin(th[0]), - self.L * cos(th[0])
            pendulum2_x, pendulum2_y = pendulum1_x - self.L * sin(th[1]), pendulum1_y - self.L * cos(th[1])

            line1.set_xdata([0, pendulum1_x])
            line1.set_ydata([0, pendulum1_y])
            line2.set_xdata([pendulum1_x, pendulum2_x])
            line2.set_ydata([pendulum1_y, pendulum2_y])
            ax.set_xlabel(label)
            return line1, line2, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.history), skip), interval=20)
        anim.save(file_name, dpi=dpi, writer='imagemagick')
        plt.show()

class Pendulum1Env(Pendulum):
    def __init__(self, x_0=0., dt=0.1):
        super().__init__(n_pend=1, x_0=x_0, dt=dt, c=0.2, g=1.0, m=1.0, L=1.0)

    def setup_equation(self, u_prime, x, torques):
        T = torques[0]
        th, u = x

        eq1 = u_prime[0] - self.g / self.L * sin(th) - T / (self.m * self.L ** 2) + self.c * u / (self.m * self.L ** 2)
        return eq1


class Pendulum2Env(Pendulum):
    def __init__(self, x_0=0., dt=0.001):
        super().__init__(n_pend=2, x_0=x_0, dt=dt)

    def setup_equation(self, u_prime, x, torques):
        T1, T2 = torques
        th1, th2, u1, u2 = x
        u1_prime, u2_prime = u_prime

        eq1 = 2 * u1_prime + u2_prime * cos(th2 - th1) - u2 ** 2 * sin(th2 - th1) \
              - self.g / self.L * sin(th1) * 2 - T1 / (self.m * self.L ** 2) + self.c * u1 / (self.m * self.L ** 2)
        eq2 = u2_prime + u1_prime * cos(th2 - th1) + u1 ** 2 * sin(th2 - th1) \
              - self.g / self.L * sin(th2) - T2 / (
                      self.m * self.L ** 2) + self.c * u2 / (self.m * self.L ** 2)
        return eq1, eq2


class Pendulum3Env(Pendulum):
    def __init__(self, x_0=0., dt=0.001):
        super().__init__(n_pend=3, x_0=x_0, dt=dt)

    def setup_equation(self, u_prime, x, torques):
        T1, T2, T3 = torques
        th1, th2, th3, u1, u2, u3 = x
        u1_prime, u2_prime, u3_prime = u_prime

        inertia = (self.m * self.L ** 2)

        eq1 = 3 * u1_prime + 2 * u2_prime * cos(th1 - th2) + u3_prime * cos(th1 - th3) \
              + 2 * u1 ** 2 * sin(th1 - th2) + u3 ** 2 * sin(th1 - th3) \
              - self.g / self.L * sin(th1) * 3 \
              - T1 / inertia + self.c * u1 / inertia
        eq2 = 2 * u1_prime * cos(th1 - th2) + 2 * u2_prime + u3_prime * cos(th2 - th3) \
              - 2 * u1 ** 2 * sin(th1 - th2) + u3 ** 2 * sin(th2 - th3) \
              - self.g / self.L * sin(th2) * 2 \
              - T2 / inertia + self.c * u2 / inertia
        eq3 = u1_prime * cos(th1 - th3) + u2_prime * cos(th2 - th3) + u3_prime \
              - u1 ** 2 * sin(th1 - th3) - u2 ** 2 * sin(th2 - th3) \
              - self.g / self.L * sin(th3) \
              - T3 / inertia + self.c * u3 / inertia

        return eq1, eq2, eq3


from gym.envs.registration import make
if __name__ == '__main__':
    p1 = Pendulum1Env(x_0=[2., 0.], dt =0.01)
    p1.reset()

    for _ in range(500):
        p1.step([0.])
        print(p1.x)
    p1.render()


    # p3 = Pendulum3Env(x_0=[3., 1., 3., 1., 0., 1.], dt=0.05)
    # p3.reset()
    #
    # for _ in range(500):
    #     p3.step([1., 0.5, 0.])
    #     print(p3.x)
    # p3.render()

    # env = make("Pendulum2-v0", dt=0.01, x_0=[1., 1.5, 0., 0.])
    # env.reset()
    # for _ in range(200):
    #     env.step([0., 0.])
    # env.render()
    # env.animate(file_name="pend2_freefal.gif",  dpi = 90)

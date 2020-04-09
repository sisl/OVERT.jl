import gym
import pickle
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CartPoleEnv2(gym.Env):
    """
    """
    """
    this is a cartpole environment
        -mp: mass of the pole
        -mc: mass of the card
        -l: length of the pole
        -x_0: initial state
        -F: force on the cart
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, mp=2.0, mc=10., l=1, g=9.8,  # physical paramters
                 x_0=[0., 0., 0., 0.], init_guess=[0.], max_force=1000,  # initializations
                 integration_method="1st", dt=0.01, N=1  # computational parameters.
                 ):
        self.mp, self.mc, self.l = mp, mc, l
        self.g, self.dt, self.N, self.max_force = g, dt, N, max_force
        self.x_0, self.init_guess = x_0, init_guess
        self.x, self.current_guess, self.history, self.viewer = [None] * 4  # these will be updated in the reset.
        self.integration_method = integration_method

        self.reset()

    def reset(self):
        self.x = np.array(self.x_0)
        self.current_guess = self.init_guess
        self.history = [{"x": self.x.copy(), "force": 0, "reward": 0}]

    def setup_equation(self, x, t, force):
        xdot = np.zeros(x.shape)
        xdot[0] = x[2]
        xdot[1] = x[3]
        # xdot[2] = (force + self.mp * sin(x[1])*(-self.l * x[3]**2 + self.g * cos(x[1])))/(self.mc + self.mp * sin(x[1])**2)
        # xdot[3] = (force*cos(x[1]) - self.mp * self.l * x[3]**2 * cos(x[1]) * sin(x[1]) + (self.mc + self.mp) * self.g * sin(x[1]))/(self.l * (self.mc + self.mp * sin(x[1])**2))
        xdot[2] = (force + self.mp * sin(x[1])*(self.l * x[3]**2 + self.g * cos(x[1])))/(self.mc + self.mp * sin(x[1])**2)
        xdot[3] = (-force*cos(x[1]) - self.mp * self.l * x[3]**2 * cos(x[1]) * sin(x[1]) - (self.mc + self.mp) * self.g * sin(x[1]))/(self.l * (self.mc + self.mp * sin(x[1])**2))

        return xdot

    def step(self, force):
        # if self.x.ndim == 1:
        #     self.x = odeint(self.setup_equation, self.x, [0, self.dt], args=(force, ))
        # else:
        #     self.x = odeint(self.setup_equation, self.x[-1, :], [0, self.dt], args=(force, ))
        if self.integration_method == "1st":
            self.x += self.setup_equation(self.x, 0, force)*self.dt
        else:
            self.x = odeint(self.setup_equation, self.x, [0., self.dt], args=(force,))[-1, :]
        self.history.append({"x": self.x.copy(), "reward": 0, "force": force})
        return self.x, 0, False, {}

    def save_history(self, name="history.pkl"):
        with open(name, 'wb') as fid:
            pickle.dump(self.history, fid, protocol=pickle.HIGHEST_PROTOCOL)

    # def render(self, mode='human'):
    #     screen_width = 600
    #     screen_height = 400
    #
    #     world_width = 4.8
    #     scale = screen_width/world_width
    #     carty = 100 # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * self.l)
    #     cartwidth = 50.0
    #     cartheight = 30.0
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
    #         axleoffset = cartheight/4.0
    #         cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
    #         pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         pole.set_color(.8,.6,.4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth/2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(.5,.5,.8)
    #         self.viewer.add_geom(self.axle)
    #         self.track = rendering.Line((0,carty), (screen_width,carty))
    #         self.track.set_color(0,0,0)
    #         self.viewer.add_geom(self.track)
    #
    #         self._pole_geom = pole
    #
    #     if self.x is None: return None
    #
    #     # Edit the pole polygon vertex
    #     pole = self._pole_geom
    #     l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
    #     pole.v = [(l,b), (l,t), (r,t), (r,b)]
    #
    #     x = self.x
    #     cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[1])
    #
    #     return self.viewer.render(return_rgb_array = mode=='rgb_array')

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None


    def render(self, mode="human", read_file=False, close=False, skip=1):

        lcart = self.l*1
        hcart = self.l*0.5
        rad_wheel = self.l*0.1
        if read_file:
            with open("history.pkl", "rb") as fid:
                self.history = pickle.load(fid)

        plt.figure(figsize=(14,6))
        for i in range(0, len(self.history), skip):
            # computing coordinates
            limits = self.l * 2
            xcart = self.history[i]["x"][0]
            # while xcart > limits:
            #     xcart -= limits


            vcart = self.history[i]["x"][2]
            ycart = 0
            thpole = self.history[i]["x"][1] - np.pi
            vpole  = self.history[i]["x"][3]
            force = self.history[i]["force"]

            xpole = xcart - self.l*sin(thpole)
            ypole = ycart + hcart + self.l*cos(thpole)
            plt.cla()

            cart = plt.Rectangle((xcart - 0.5*lcart, ycart ), lcart, hcart)
            rwheel = plt.Circle((xcart + 0.2*lcart, ycart-rad_wheel), rad_wheel)
            lwheel = plt.Circle((xcart - 0.2*lcart, ycart-rad_wheel), rad_wheel)
            plt.plot((xcart, xpole), (ycart + hcart, ypole), linewidth=5, color='r')
            plt.gca().add_patch(cart)
            plt.gca().add_patch(rwheel)
            plt.gca().add_patch(lwheel)


            plt.xlim([-limits*3, limits*3])
            plt.ylim([-limits*0.2, limits*0.75])
            plt.gca().set_aspect('equal', 'box')
            # typing state inputs.

            xtext, ytest = 0.1 * limits, 0.8 * limits
            plt.text(xtext, ytest, "x:%0.1f, th:%0.1f, u:%0.1f, w:%0.1f" % (xcart, thpole, vcart, vpole))
            xtext, ytest = 0.1 * limits, 0.65 * limits
            plt.text(xtext, ytest, "force= %0.2f" % force)
            plt.draw()
            plt.pause(0.01)
        if close:
            plt.close()
    #
    # def animate(self, skip=1):
    #     fig, ax = plt.subplots()
    #     fig.set_tight_layout(True)
    #
    #     # Plot a scatter that persists (isn't redrawn) and the initial line.
    #     x = self.history[0]["x"]
    #     th = x[:2]
    #     pendulum1_x, pendulum1_y = - self.L * sin(th[0]), - self.L * cos(th[0])
    #     pendulum2_x, pendulum2_y = pendulum1_x - self.L * sin(th[1]), pendulum1_y - self.L * cos(th[1])
    #
    #     line1, = ax.plot([0, pendulum1_x], [0, pendulum1_y], color='black')
    #     line2, = ax.plot([pendulum1_x, pendulum2_x], [pendulum1_y, pendulum2_y], color='black', markersize=10,
    #                      markerfacecolor='red', marker="o")
    #     limits = self.L * 2.2
    #     plt.xlim([-limits, limits])
    #     plt.ylim([-limits, limits])
    #
    #     def update(i):
    #         label = 'timestep {0}'.format(i)
    #         print(label)
    #         x = self.history[i]["x"]
    #         th = x[:2]
    #         u = x[2:]
    #         reward = self.history[i]["reward"]
    #         torque = self.history[i]["torques"]
    #
    #         pendulum1_x, pendulum1_y = - self.L * sin(th[0]), - self.L * cos(th[0])
    #         pendulum2_x, pendulum2_y = pendulum1_x - self.L * sin(th[1]), pendulum1_y - self.L * cos(th[1])
    #
    #         line1.set_xdata([0, pendulum1_x])
    #         line1.set_ydata([0, pendulum1_y])
    #         line2.set_xdata([pendulum1_x, pendulum2_x])
    #         line2.set_ydata([pendulum1_y, pendulum2_y])
    #         ax.set_xlabel(label)
    #         return line1, line2, ax
    #
    #     anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.history), skip), interval=50)
    #     anim.save('line.gif', dpi=80, writer='imagemagick')
    #     plt.show()


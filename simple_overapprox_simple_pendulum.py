import numpy as np
from numpy import array
import matplotlib.pyplot as plt
#import sympy as sp

class line():
    def __init__(self,*args, init_type='two points', **kwargs):
        if init_type == 'two points':
            self.p1 = kwargs.get('p1')
            self.p2 = kwargs.get('p2')
            self.m = self.get_slope()
        elif init_type == 'point slope':
            self.p1 = kwargs.get('p1')
            self.m = kwargs.get('m')
        self.b = self.get_intercept()
        self.eq = lambda x : self.m*x + self.b
    def get_slope(self):
        # get slop using two points
        x1,y1 = self.p1
        x2,y2 = self.p2
        return (y2-y1)/(x2-x1)
    def get_intercept(self):
        x1,y1 = self.p1
        return -self.m*x1 + y1

# pass in a function
def build_sin_approx(fun, c1, c2, convex_reg, concave_reg):
    # identify regions of convexity and concavity
    # for c1*sin(x) + c2, constant concavity from -180 to 0 (convex)
    # and constant concavity from 0 to 180
    max_lip = c1
    offset = c2
    #convex_reg = [-np.pi, 0.0]
    #concave_reg = [0.0, np.pi]
    # construct upper and lower bounds accordingly
    #
    # given a function and an interval, construct a "max lipshitz" bound
    # flag should either be LB or UP
    # max lipshitz is used to LB of convex regions and UB of concave regions
    def construct_max_lip_bound(fun, interval, flag=None, max_lip=None):
        # segment into increasing portion and decreasing portion
        # luckily, for sin functions, just divide interval in two
        reg1 = [interval[0], (interval[1] - interval[0])/2.0 + interval[0]]
        reg2 = [reg1[1], interval[1]]
        min_lip = -1.0*max_lip
        if flag == "UB": # this means we are constructing the UB of a concave region
            dec = reg2
            inc = reg1
            # define lines by two points
            strt_pt = fun(inc[0])
            inc_len = inc[1] - inc[0]
            line1 = np.array([[inc[0], strt_pt], [inc[1], strt_pt + max_lip*inc_len]])
            dec_len = dec[1] - dec[0]
            line2 = np.array([line1[1],[dec[1],line1[1][1] + min_lip*dec_len]])
        elif flag == "LB": # this means we are constructing the LB of a convex region
            # work backwards for this region
            dec = reg1
            inc = reg2
            strt_pt = fun(inc[1])
            inc_len = inc[1] - inc[0]
            line2 = np.array([[inc[0], strt_pt - max_lip*inc_len],[inc[1], strt_pt]])
            dec_len = dec[1] - dec[0]
            line1 = np.array([[dec[0], line2[0][1] - min_lip*dec_len ],line2[0]])
        #
        return (line1, line2)
    #
    def construct_jensen_bound(fun, interval):
        reg1 = [interval[0], (interval[1] - interval[0])/2.0 + interval[0]]
        reg2 = [reg1[1], interval[1]]
        import pdb; pdb.set_trace()
        line1 = np.array([[reg1[0], fun(reg1[0]) ], [reg1[1], fun(reg1[1]) ]])
        line2 = np.array([line1[1], [reg2[1], fun(reg2[1])] ])
        #
        return (line1, line2)
    #
    # for (convex) part, construct max lip LB and jensen upper bound
    cvx_LB = construct_max_lip_bound(fun, convex_reg, flag = "LB", max_lip=max_lip)
    cvx_UB = construct_jensen_bound(fun, convex_reg)
    # for (concave) part, construct jensen lower bound and max lip upper bound
    ccv_LB = construct_jensen_bound(fun, concave_reg)
    ccv_UB = construct_max_lip_bound(fun, concave_reg, flag="UB", max_lip=max_lip)
    #
    # can do the offset at the very end (maybe the scaling too? oh well)
    return ((cvx_LB, ccv_LB),(cvx_UB, ccv_UB))


xdat =  np.linspace(-np.pi, np.pi, 100)
fun = lambda x: -5*np.sin(x) + 5
sindat = fun(xdat)

# build_sin_approx(fun, c1, c2, convex_reg, concave_reg)
LB, UB = build_sin_approx(fun, 5, None, [0, np.pi], [-np.pi, 0])

plt.plot(xdat, sindat)

for b in [LB,UB]:
    for item in b:
        for line in item:
            plt.plot(line[:,0], line[:,1])

plt.show()

# turn each set of two points into an equation





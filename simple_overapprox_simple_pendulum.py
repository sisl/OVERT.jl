import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import tensorflow as tf
#import sympy as sp

# turn max and min into relus
def ReLu(x):
    return np.maximum(x,0.0)

def ReLuMax(x,y):
    return ReLu(x-y) + y

def tfReLuMax(x,y, name=""):
    return tf.add(tf.nn.relu(x-y), y, name=name+"tfrelu_0")

def ReLuMin(x,y):
    return -ReLuMax(-x,-y)

def tfReLuMin(x,y, name=""):
    return tf.multiply(-1.0, tfReLuMax(-x,-y), name=name+"tfrelu_1")

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
        with tf.name_scope("line"):
            self.tfm = tf.constant([self.m])
            self.tfb = tf.constant([self.b])
            self.tfeq = lambda x: self.tfm*x + self.tfb
    def get_slope(self):
        # get slop using two points
        x1,y1 = self.p1
        x2,y2 = self.p2
        return (y2-y1)/(x2-x1)
    def get_intercept(self):
        x1,y1 = self.p1
        return -self.m*x1 + y1

class bound():
    def __init__(self, line1, line2, combi_op, meta_combi_op, meta_combi_const):
        self.line1 = line1
        self.line2 = line2
        self.combi_op = combi_op
        self.meta_combi_op = meta_combi_op
        self.meta_combi_const = meta_combi_const
        self.meta_combi_const_tf = tf.constant(self.meta_combi_const)
        self.eval_bound = lambda x: meta_combi_op(combi_op(line1.eq(x), line2.eq(x)), meta_combi_const)
        # tensorflow stuff
        if combi_op == np.minimum:
            self.combi_op_tf = tfReLuMin
        elif combi_op == np.maximum:
            self.combi_op_tf = tfReLuMax
        # 
        if meta_combi_op == np.minimum:
            self.meta_combi_op_tf = tfReLuMin
        elif meta_combi_op == np.maximum:
            self.meta_combi_op_tf = tfReLuMax
        # eval bound, in tf
        with tf.name_scope("bound"):
            self.tf_eval_bound = lambda x: self.meta_combi_op_tf(
                self.combi_op_tf(line1.tfeq(x), line2.tfeq(x)), 
                self.meta_combi_const_tf)

# pass in a function
def build_sin_approx(fun, c1, convex_reg, concave_reg):
    # identify regions of convexity and concavity
    # for c1*sin(x) + c2, constant concavity from -180 to 0 (convex)
    # and constant concavity from 0 to 180
    max_lip = c1
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
        return (line(init_type='two points', p1=line1[0,:], p2=line1[1,:]), line(init_type='two points', p1=line2[0,:], p2=line2[1,:]))
    #
    def construct_jensen_bound(fun, interval):
        reg1 = [interval[0], (interval[1] - interval[0])/2.0 + interval[0]]
        reg2 = [reg1[1], interval[1]]
        #import pdb; pdb.set_trace()
        line1 = np.array([[reg1[0], fun(reg1[0]) ], [reg1[1], fun(reg1[1]) ]])
        line2 = np.array([line1[1], [reg2[1], fun(reg2[1])] ])
        #
        return (line(init_type='two points', p1=line1[0,:], p2=line1[1,:]), line(init_type='two points', p1=line2[0,:], p2=line2[1,:]))
    #
    # for (convex) part, construct max lip LB and jensen upper bound
    cvx_LB_lines = construct_max_lip_bound(fun, convex_reg, flag = "LB", max_lip=max_lip)
    cvx_LB = bound(cvx_LB_lines[0], cvx_LB_lines[1], np.maximum, np.minimum, 0.0)
    cvx_UB_lines = construct_jensen_bound(fun, convex_reg)
    cvx_UB = bound(cvx_UB_lines[0], cvx_UB_lines[1], np.maximum, np.minimum, 0.0)
    # for (concave) part, construct jensen lower bound and max lip upper bound
    ccv_LB_lines = construct_jensen_bound(fun, concave_reg)
    ccv_LB = bound(ccv_LB_lines[0], ccv_LB_lines[1], np.minimum, np.maximum, 0.0)
    ccv_UB_lines = construct_max_lip_bound(fun, concave_reg, flag="UB", max_lip=max_lip)
    ccv_UB = bound(ccv_UB_lines[0], ccv_UB_lines[1], np.minimum, np.maximum, 0.0)
    #
    # can do the offset at the very end (maybe the scaling too? oh well)
    return ((cvx_LB, ccv_LB),(cvx_UB, ccv_UB))

def testing():
    xdat =  np.linspace(-np.pi, np.pi, 200)
    c1=2.0
    c2=-5.0
    fun = lambda x: c1*np.sin(x)
    sindat = fun(xdat)

    # build_sin_approx(fun, c1, convex_reg, concave_reg)
    LB, UB = build_sin_approx(fun, c1, [-np.pi, 0], [0, np.pi])

    plt.plot(xdat, sindat+c2)

    plt.plot(xdat, LB[0].eval_bound(xdat) + LB[1].eval_bound(xdat) + c2)
    plt.plot(xdat, UB[0].eval_bound(xdat) + UB[1].eval_bound(xdat) + c2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str(c1)+'*Sin(x) + '+str(c2))
    plt.show()

    # swap maxes and mins for ReLuMax and ReLuMin
    def swap_out_for_relus(op):
        if op == np.maximum:
            op = ReLuMax
        elif op == np.minimum:
            op = ReLuMin
        return op

    for b in LB:
        b.combi_op = swap_out_for_relus(b.combi_op)
        b.meta_combi_op = swap_out_for_relus(b.meta_combi_op)

    for b in UB:
        b.combi_op = swap_out_for_relus(b.combi_op)
        b.meta_combi_op = swap_out_for_relus(b.meta_combi_op)

    plt.plot(xdat, sindat+c2)

    plt.plot(xdat, LB[0].eval_bound(xdat) + LB[1].eval_bound(xdat) + c2)
    plt.plot(xdat, UB[0].eval_bound(xdat) + UB[1].eval_bound(xdat) + c2)

    plt.show()

    print("LB: ", LB[0].eval_bound(1.0) + LB[1].eval_bound(1.0) + c2)
    print("UB: ", UB[0].eval_bound(1.0) + UB[1].eval_bound(1.0) + c2)

    # woo! The ReLu Maxes are implemented correctly :)

    #######################################################################

    # build a graph where there is a single scalar, theta, as input
    theta = tf.Variable([1.0], name="theta")

    # meta_combi_op(combi_op(line1.eq(x), line2.eq(x)), meta_combi_const)
    with tf.name_scope("LB"):
        tfLB = LB[0].tf_eval_bound(theta) + LB[1].tf_eval_bound(theta) + tf.constant([c2])
    with tf.name_scope("UB"):
        tfUB = UB[0].tf_eval_bound(theta) + UB[1].tf_eval_bound(theta) + tf.constant([c2])

    # test the graph so far
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([tfLB, tfUB])

    print("tfLB: ", sess.run(tfLB))
    print("tfUB: ", sess.run(tfUB))

    # whoo! The tf implementation works!

    # Now.....visualize the graph in tensorboard?
    LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/n6"
    train_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    train_writer.add_graph(sess.graph)
    train_writer.close()

# then do the euler integration piece
# testing()
















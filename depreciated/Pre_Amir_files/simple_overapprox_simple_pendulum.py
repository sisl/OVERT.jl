import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import tensorflow as tf
#import sympy as sp
from relu_approximations import ReLu, ReLuMax, tfReLuMax, ReLuMin, tfReLuMin, min_from_max

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
        # get slope using two points
        x1,y1 = self.p1
        x2,y2 = self.p2
        return (y2-y1)/(x2-x1)
    def get_intercept(self):
        x1,y1 = self.p1
        return -self.m*x1 + y1

"""
A class for containing the two-piece upper or lower bound for a single convex or concave region.
"""
class bound():
    def __init__(self, line1, line2, combi_op, meta_combi_op, meta_combi_const, use_truemax=True):
        self.line1 = line1
        self.line2 = line2
        self.combi_op = combi_op
        self.meta_combi_op = meta_combi_op
        self.meta_combi_const = meta_combi_const
        self.meta_combi_const_tf = tf.constant(self.meta_combi_const)
        self.eval_bound = lambda x: meta_combi_op(combi_op(line1.eq(x), line2.eq(x)), meta_combi_const)
        # tensorflow stuff
        combis = []
        for op in [combi_op, meta_combi_op]:
            if op == np.minimum:
                if use_truemax:
                    combis.append(min_from_max)
                else:
                    combis.append(tfReLuMin)
            elif op == np.maximum:
                if use_truemax:
                    combis.append(tf.maximum)
                else:
                    combis.append(tfReLuMax)
        self.combi_op_tf = combis[0]
        self.meta_combi_op_tf = combis[1]

    # eval bound, in tf
    def tf_apply_bound(self, x):
        with tf.name_scope("bound"):
            a = self.meta_combi_op_tf is tfReLuMax
            b = self.meta_combi_op_tf is tf.maximum
            c = self.meta_combi_const == 0.0
            d = self.meta_combi_op_tf is tfReLuMin
            e = self.meta_combi_op_tf is tf.minimum
            f = self.meta_combi_op_tf is min_from_max
            if ((a or b) and c): # max of a value and 0 is just a relu
                return tf.nn.relu(
                    self.combi_op_tf(self.line1.tfeq(x), self.line2.tfeq(x))
                    )
            elif ((d or e or f) and c): # min of a value and 0 is -relu(-x)
                return -1*tf.nn.relu(
                    -1*self.combi_op_tf(self.line1.tfeq(x), self.line2.tfeq(x))
                    )
            else:
                return self.meta_combi_op_tf(
                self.combi_op_tf(self.line1.tfeq(x), self.line2.tfeq(x)), 
                self.meta_combi_const_tf)

"""
A class where all bound pieces can be added together to create a single LB object or single UB object.
"""
class additive_bound():
    def __init__(self, pieces):
        self.pieces = pieces # pieces of bounds that will be added together
    def eval(self, x): # additive evaluation
        out = self.pieces[0].eval_bound(x)
        for i in range(1, len(self.pieces)):
            out = out + self.pieces[i].eval_bound(x)
        return out
    def tf_apply(self, x):
        with tf.name_scope("additive_bound"):
            out = self.pieces[0].tf_apply_bound(x)
            for i in range(1, len(self.pieces)):
                out = out + self.pieces[i].tf_apply_bound(x)
            return out

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
    LB = additive_bound([cvx_LB, ccv_LB])
    UB = additive_bound([cvx_UB, ccv_UB])
    return (LB, UB)

def testing():
    xdat =  np.linspace(-np.pi, np.pi, 200)
    c1=5.0
    c2=-5.0
    fun = lambda x: c1*np.sin(x)
    sindat = fun(xdat)

    # build_sin_approx(fun, c1, convex_reg, concave_reg)
    LB, UB = build_sin_approx(fun, c1, [-np.pi, 0], [0, np.pi])

    plt.plot(xdat, sindat+c2)

    plt.plot(xdat, LB.eval(xdat) + c2)
    plt.plot(xdat, UB.eval(xdat) + c2)

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

    for b in LB.pieces:
        b.combi_op = swap_out_for_relus(b.combi_op)
        b.meta_combi_op = swap_out_for_relus(b.meta_combi_op)

    for b in UB.pieces:
        b.combi_op = swap_out_for_relus(b.combi_op)
        b.meta_combi_op = swap_out_for_relus(b.meta_combi_op)

    plt.plot(xdat, sindat+c2)
    #
    plt.plot(xdat, LB.eval(xdat) + c2)
    plt.plot(xdat, UB.eval(xdat) + c2)

    plt.figure()
    plt.plot(xdat, sindat - LB.eval(xdat), label="sin - lb" )
    plt.plot(xdat, UB.eval(xdat) -sindat, label="ub - sin")
    plt.plot(xdat, UB.eval(xdat) - LB.eval(xdat), label="ub - lb")

    plt.legend()
    plt.show()

    print("LB: ", LB.eval(1.0) + c2)
    print("UB: ", UB.eval(1.0) + c2)

    # woo! The ReLu Maxes are implemented correctly :)

    #######################################################################

    # build a graph where there is a single scalar, theta, as input
    theta = tf.placeholder(shape=(1,), name="theta", dtype='float32')

    # meta_combi_op(combi_op(line1.eq(x), line2.eq(x)), meta_combi_const)
    with tf.name_scope("LB"):
        tfLB = LB.tf_apply(theta) + tf.constant([c2])
    with tf.name_scope("UB"):
        tfUB = UB.tf_apply(theta) + tf.constant([c2])

    # test the graph so far
    sess = tf.Session()
    [tfLB_v, tfUB_v] = sess.run([tfLB, tfUB], feed_dict={theta: np.array([1.0])})
    print("tfLB: ", tfLB_v)
    print("tfUB: ", tfUB_v)

    # eval at all the points
    lb = []; ub = []
    for x in xdat:
        lb.append(sess.run([tfLB], feed_dict={theta: np.array([x])}))
        ub.append(sess.run([tfUB], feed_dict={theta: np.array([x])}))

    lb = np.array(lb).flatten()
    ub = np.array(ub).flatten()

    # plot
    plt.plot(xdat, sindat + c2)
    plt.plot(xdat, lb)
    plt.plot(xdat, ub)
    plt.show()

    # whoo! The tf implementation works!

    # Now.....visualize the graph in tensorboard?
    LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/n6"
    train_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    train_writer.add_graph(sess.graph)
    train_writer.close()

def testing2():
    # seeing how many assumptions were made and if it works for anything
    # other than sine. looks like it doesn't
    def fun(x):
        if x > .2:
            return np.sin(x) + np.log(x)
        else:
            return np.sin(x) + np.log(0.2)
    xdat =  np.linspace(-np.pi, np.pi, 200)
    c1= fun(0.2)
    c2= np.log(0.2)
    sindat = [fun(x) for x in xdat]

    # build_sin_approx(fun, c1, convex_reg, concave_reg)
    LB, UB = build_sin_approx(fun, c1, [-np.pi, 0.2], [0.2, np.pi])

    plt.plot(xdat, sindat+c2)

    plt.plot(xdat, LB.eval(xdat) + c2)
    plt.plot(xdat, UB.eval(xdat) + c2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str(c1)+'*Sin(x) + '+str(c2))
    plt.show()


# then do the euler integration piece
# testing()
















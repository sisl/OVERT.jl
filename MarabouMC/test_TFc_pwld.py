# test TF network controller and pwl plant

import tensorflow as tf # 1.x
import numpy as np
from transition_systems import TFController, Dynamics, TFControlledTransitionRelation, TransitionSystem
from tf_utils import smoosh_to_const
from MC_constraints import Constraint, ConstraintType, ReluConstraint, Monomial
from marabou_interface import MarabouWrapper
from properties import ConstraintProperty
from MC_interface import BMC

# create random network controller with 2d state space and 1d control
sess = tf.Session()
with sess.as_default():
    x = tf.placeholder(shape=(2,1), dtype='float64')
    W1 = np.random.rand(2,2)
    b1 = np.random.rand(2,1)
    output = tf.nn.relu(tf.matmul(W1,x) + b1)
    W2 =  np.random.rand(1,2)
    b2 =  np.random.rand(1,1)
    output = tf.nn.relu(tf.matmul(W2,output) + b2)
    sess.run(tf.global_variables_initializer()) # actually sets Variable values to values specified

# smoosh all tf.Variables to tf.Constants, put into new graph
new_graph = smoosh_to_const(sess, output.op.name)
# create controller object with network
controller = TFController(tf_sess=tf.Session(graph=new_graph), inputNames=[x.op.name], outputName=output.op.name)

# create a super simple plant directly using constraint objects
dynamics = Dynamics(states=np.array([["x"], ["y"]]), controls=["u"], fun=np.sin)
# x' = relu(x + u)   ->   x + u - z = 0 , x' = relu(z)
c1 = Constraint(ConstraintType('EQUALITY'))
c1.monomials = [Monomial(1, "x"), Monomial(1,"u"), Monomial(-1,"z")]
c3 = ReluConstraint(varin="z", varout="x'")
# y' = y  ->  y - y' = 0
c2 = Constraint(ConstraintType('EQUALITY'))
c2.monomials = [Monomial(1,"y"), Monomial(-1, "y'")]
dynamics.constraints = [c1,c2,c3]

# create transition relation using controller and dynamics
tr = TFControlledTransitionRelation(dynamics_obj=dynamics, 
                                        controller_obj=controller)

# initial set
init_set = {"x": (1.1,2), "y": (-1,1)}

# build the transition system as an (S, I(S), TR) tuple
ts = TransitionSystem(states=tr.states, initial_set=init_set, transition_relation=tr)

# solver
solver = MarabouWrapper()

# property
p = Constraint(ConstraintType('GREATER'))
# x > c (complement will be x <= c)
p.monomials = [Monomial(1, "x")]
p.scalar = 1. # 0 #
prop = ConstraintProperty([p], ["x"])

# algo
algo = BMC(ts = ts, prop = prop, solver=solver)
algo.check_invariant_until(3)

# random runs to give intuition to MC result
for i in range(10):
    x = np.random.rand()*(2 - 1.1) + 1.1
    print("x@0=", x)
    y = np.random.rand()*(1 - -1) + -1
    for j in range(3):    
        state = np.array([x,y]).flatten().reshape(-1,1)
        u = np.maximum(0, W2@(np.maximum(0,W1@state + b1)) + b2)
        #x' = relu(x + u)
        x = max(0, x + u.flatten()[0])
        print("x@",j+1,"=", x)
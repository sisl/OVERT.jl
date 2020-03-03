# test TF network controller and pwl plant
import tensorflow as tf # 1.x
import numpy as np
from transition_systems import TFController, Dynamics, TFControlledTransitionRelation
from tf_utils import smoosh_to_const

# create random network controller with 2d state space and 1d control
sess = tf.Session()
with sess.as_default():
    x = tf.placeholder(shape=(2,1), dtype='float64')
    W = np.random.rand(1,2)
    b = np.random.rand(1,1)
    output = tf.nn.relu(tf.matmul(W,x) + b)
    sess.run(tf.global_variables_initializer()) # actually sets Variable values to values specified

# smoosh all tf.Variables to tf.Constants, put into new graph
new_graph = smoosh_to_const(sess, output.op.name)
# create controller object with network
controller = TFController(tf_sess=tf.Session(graph=new_graph), inputNames=[x.op.name], outputName=output.op.name)

# create a super simple plant directly using constraint objects
dynamics = Dynamics(np.sin, np.array([["x"], ["y"]]), ["u"])
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
p.scalar = -1. # 0 #
prop = ConstraintProperty([p])

# algo
algo = BMC(ts = ts, prop = prop, solver=solver)
algo.check_invariant_until(3)

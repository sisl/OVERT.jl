import tensorflow as tf
from gym.envs.classic_control.my_pendulum import MyPendulumEnv

# construct pendulum
env =  MyPendulumEnv()

# construct  network inputs
with tf.namescope('init'):
    theta  = tf.placeholder(shape=(1,),name='theta', dtype='float64')
    theta_dot = tf.placeholder(shape=(1,), name='theta_dot', dtype='float64')

# see if it works together
def apply_fake_sin(x):
    with tf.name_scope('fake_sin'):
        theta, theta_dot = x
        return tf.nn.relu(theta) - tf.nn.relu(-theta)
with tf.name_scope('fake_control'):
    fake_control = -2*theta - .3*theta_dot

x = [theta, theta_dot]

with tf.name_scope('apply dynamics'):
    state_tp1 = env.dynamics(x=x,u=fake_control,nonlinear_f=apply_fake_sin)

with tf.Session() as sess:
    out = sess.run([state_tp1], feed_dict={theta: [0.1], theta_dot: [0.01]})
    print(out)


# woooo! I can construct the dynamics network like this!! directly using a python function that is not explicitly written
# in tensorflow



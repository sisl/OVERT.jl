import tensorflow as tf
import numpy as np
import colored_traceback.always

def graph1():
    x = tf.Variable([[0.0], [0.0]])
    y = tf.Variable([[1.0], [1.0]])
    W1 = tf.constant([[2.0, 0.0],[0.0, 2.0]])
    b1 = tf.constant([[-1.0], [-1.0]])
    W2 = -W1
    b2 = -b1
    with tf.name_scope("matmul_1"):
        z1 = W1@x + b1
    with tf.name_scope("matmul_2"):
        z2 = W2@y + b2
    return z1,z2

def network_to_test_case_4(sess):
    R = tf.Variable([[1.0],[1.0]])
    C = tf.constant([[-1.0], [-1.0]])
    W1 = tf.constant(2*np.eye(2), dtype='float32')
    b1 = tf.constant([[5.0], [5.0]])
    W2 = tf.constant(3*np.eye(2), dtype='float32')
    b2 = tf.constant([[7.0], [7.0]])

    Q = R + C
    x= W1@Q + b1
    y = W2@R + b2
    F = x + y

    sess.run(tf.global_variables_initializer())

    correct_mats = []
    correct_mats.append(np.hstack([np.eye(2), np.eye(2)]))
    correct_mats.append(np.vstack([
            np.hstack([np.eye(2), np.zeros((2,2))
                ]),
            np.hstack([np.zeros((2,2)), np.eye(2)
                ])
        ]))
    with sess.as_default():
        correct_mats.append(np.vstack([
                            np.hstack([
                                W1.eval(),
                                np.zeros((2,2))
                                ]), 
                            np.hstack([
                                np.zeros((2,2)),
                                W2.eval()
                                ])
                            ])
                        )
    correct_mats.append(np.vstack([
                            np.hstack([np.eye(2), np.zeros((2,2)) ]),
                            np.hstack([np.zeros((2,2)), np.eye(2)])
                                  ])
                       )
    correct_mats.append(np.vstack(
                            [np.eye(2),np.eye(2)]
        )
    )
    with sess.as_default():
        correct_biases = [
            np.zeros((2,1)),
            # the add
            np.vstack([
                b1.eval(),
                b2.eval()
                ]),
            # the multiply
            np.zeros((4,1)),
            # the constant add
            np.vstack([
                C.eval(),
                np.zeros((2,1))
                ]),
            np.zeros((4,1))

        ]
    return F, correct_mats, correct_biases

# handle_duplicates_immediately_before_relus
def network_to_test_case_3(sess):
    R = tf.Variable([[1.0],[1.0]])
    A = tf.nn.relu(R)
    B = tf.nn.relu(R)

    sess.run(tf.global_variables_initializer())

    correct_mat = [np.vstack([
            np.eye(2),
            np.eye(2)
        ])
    ]
    correct_bias = [np.zeros((4,1))]

    return [A,B], correct_mat, correct_bias

def network_to_test_case_1(sess):
    R = tf.Variable([[1.0],[1.0]])
    W = tf.constant(np.random.rand(2,2), dtype='float32')
    x = W@R
    A = tf.nn.relu(x)
    B = tf.nn.relu(x)

    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        correct_mats = [
            np.vstack([
                    np.eye(2),
                    np.eye(2)
                ]),
            W.eval(),
        ]
    correct_bias = [
        np.zeros((4,1)),
        np.zeros((2,1))
    ]
    return ([A,B], correct_mats, correct_bias)

# AKA duplication preceding a mix of variables, activations, and matmuls
def network_to_handle_duplicates_prior_to_mix():
    pass

########################################################
sess = tf.Session()

# testing zone
#Output, M, B = network_to_test_case_4(sess)

#Outputs, M, B = network_to_test_case_3(sess)

Outputs, M, B = network_to_test_case_1(sess)
print("M: ", M)
print("B: ", B)


def write(sess):
    LOGDIR = "/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/tensorboard_logs/simple_parsing"
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    train_writer.close()

import parsing as p

fW, fb = p.parse_network([o.op for o in Outputs], [], [], [], [], 'Relu', sess)








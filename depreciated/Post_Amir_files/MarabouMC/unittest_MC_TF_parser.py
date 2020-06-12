################################################################
## This is for tf 2.x
## Because many RL frameworks don't yet use tf 2.x, I am abandoning this
## Feb. 17 2020
## Chelsea Sidrane
################################


import tensorflow as tf # 2.0
import numpy as np
from write_graphdef import write_graphdef
from MC_TF_parser import TFConstraint
import os

# create network
class CustomModel(tf.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.W = np.random.rand(3,5)
        self.b = np.random.rand(3,1)
        #
    @tf.function
    def __call__(self, x):
        return tf.nn.relu(tf.matmul(self.W,x) + self.b)

model = CustomModel()
print(model(tf.constant(np.random.rand(5,1))))
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, 'tfpb/')):
    os.mkdir(os.path.join(cwd, 'tfpb/'))
f_id = str(int(np.round(np.random.rand()*5000)))
filedir = "tfpb/model_"+f_id
tf.saved_model.save(model, filedir)

# read graphdef
#tfconstraints = TFConstraint(filename, inputNames=[x.op.name], outputName=out.op.name)


# test that output of tf network satisfies constraints of TFConstraint